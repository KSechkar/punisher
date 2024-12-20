'''
POP_SIMULATOR.PY: a class for simulating populations of cells with the punisher and a burdensome gene
'''
# By Kirill Sechkar

# NOTE: As opposed to the notation used in the paper, here we maintain a "one gene functionality=one character" rule,
# so that the strings describing gene functionalities are always a consistent length, which simplifies the code.
# Thus, a non-functional gene "G" is denoted by "O" in its place, rather than "G". For example, the cells with
# all genes functional, save for the mutated burdensome gene B, are denoted as "B'SPC" in the paper and as "OSPC" here.

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import functools
import itertools
from diffrax import diffeqsolve, Heun, ODETerm, SaveAt, PIDController, SteadyStateEvent
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, palettes as bkpalettes, transform as bktransform
from math import pi
import time

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
import synthetic_circuits as circuits
from cell_model import *


# POPULATION SIMULATION
class PopulationSimulator:
    # initialise
    def __init__(self,
                 mutation_rates=None, # gene mutation rates
                 transition_rates=None, # switch state transition rates
                 integrase_activities=None, # integrase activity
                 growth_rates=None, # growth rates in all states
                 p_bs=None, # burdensome protein concentrations
                 ):
        # dictionary: cell state descriptions vs position in array for ODE simulation
        # (0/1,0/1,0/1,0/1) - presence of functional burden, switch, integrase and CAT genes
        # L/H/0/D - low level, high level or no switch protein; D for 'dying', i.e. growth disabled by the integrase
        self.cs2p={}
        array_ind=0
        for func_bsi in itertools.product([0,1],repeat=4):
            self.cs2p[func_bsi]={'0':array_ind, 'L':array_ind+1, 'H':array_ind+2}
            array_ind+=3

        # record total number of possible cell states
        self.num_css=array_ind

        # match tuples describing gene functionalities to character strings
        self.func2string, self.string2func=self.fill_func2string_and_string2func()

        # get legend labels - in the code style and in the paper style (see the NOTE above)
        self.legend_labels=self.make_legend_labels()

        # mutation rates (per 1/h growth rate)
        if(mutation_rates!=None):
            self.mutation_rates=mutation_rates
        else:
            self.mutation_rates=[
                0.1,    # burden
                0.1,    # switch/integrase
                0.1,    # protease
                0.1]    # CAT

        # switch state transition rates (1/h units)
        self.transition_rates={}
        for func in self.cs2p.keys():
            self.transition_rates[func]={}
            for state in self.cs2p[func].keys():
                self.transition_rates[func][state]={}
                for to_state in self.cs2p[func].keys():
                    self.transition_rates[func][state][to_state]=0
        if(transition_rates!=None):
            for func_string in transition_rates.keys():
                for state_tostate in transition_rates[func_string].keys():
                    state=state_tostate.split('>')[0]
                    to_state=state_tostate.split('>')[1]
                    self.transition_rates[self.string2func[func_string]][state][to_state]=transition_rates[func_string][state_tostate]

        # integrase activity
        if(integrase_activities!=None):
            self.integrase_activities=integrase_activities
        else:
            # action rate depends on growth rate (i.e. presence of burden) and switch state
            self.integrase_activities={'B': {'H':3, 'L':1e-4, '0':0},
                                       'O': {'H':3, 'L':1e-4, '0':0}}

        # growth rates in all states - a dictionary and a vector
        self.growth_rates = {}
        self.grv = np.zeros(self.num_css)
        if (growth_rates!=None):
            for func in self.cs2p.keys():
                self.growth_rates[func] = {}
                for state in self.cs2p[func].keys():
                    self.growth_rates[func][state] = growth_rates[self.func2string[func]][state]
                    self.grv[self.cs2p[func][state]] = growth_rates[self.func2string[func]][state]
        else:
            for func in self.cs2p.keys():
                self.growth_rates[func] = {}
                for state in self.cs2p[func].keys():
                    self.growth_rates[func][state] = 1
                    self.grv[self.cs2p[func][state]] = 1
        # cell division rate dictionary and vector - growth rates divided by ln(2) to account for cell doubling times
        self.division_rates = {}
        for func in self.cs2p.keys():
            self.division_rates[func] = {}
            for state in self.cs2p[func].keys():
                self.division_rates[func][state] = self.growth_rates[func][state]/np.log(2)
        self.drv = self.grv/np.log(2)

        # burdensome protein concentrations - in  a vector
        self.pxv = np.zeros(self.num_css)
        if(p_bs!=None):
            for func in self.cs2p.keys():
                for state in self.cs2p[func].keys():
                    self.pxv[self.cs2p[func][state]] = p_bs[self.func2string[func]][state]
        else:
            for func in self.cs2p.keys():
                for state in self.cs2p[func].keys():
                    self.pxv[self.cs2p[func][state]] = 0

        # get the matrix of cell division rates (a divided cell is considered non-existent - but it will does produce two daughters)
        self.drm=self.fill_division_matrix()

        # get the matrix of cell birth rates
        self.brm=self.fill_birth_matrix()

        # get the matrix of switch state transition rates
        self.trm=self.fill_transition_matrix()

        # get the matix of integrase action transitions
        self.irm=self.fill_integrase_matrix()

        return

    # match tuples describing gene functionalities to character strings and back
    def fill_func2string_and_string2func(self):
        func2string={}
        for func in self.cs2p.keys():
            func2string[func]=''
            for gene in range(0,len(func)):
                if(func[gene]==1):
                    func2string[func]+='BSPC'[gene]
                else:
                    func2string[func]+='O'

        string2func={}
        for func in self.cs2p.keys():
            string2func[func2string[func]]=func

        return func2string, string2func

    # get legend labels - in the code style and in the paper style (see the NOTE above)
    def make_legend_labels(self):
        legend_labels={'code':self.func2string, 'paper':{}}
        for func in self.cs2p.keys():
            legend_labels['paper'][func]=''
            for gene in range(0,len(func)):
                legend_labels['paper'][func]+='BSPC'[gene]
                if(func[gene]==0):
                    legend_labels['paper'][func]+='\''
        return legend_labels

    # FILLING MATRICES
    # all in format [to,from]
    # fill the matrix of cell division rates = growth rates/ln(2)
    def fill_division_matrix(self):
        # initialise the division rate matrix
        drm=np.zeros((self.num_css,self.num_css))
        # just get a diagonal matrix with the growth rate of a given mutation-switch cell state
        for func in self.cs2p.keys():
            for state in self.cs2p[func].keys():
                drm[self.cs2p[func][state], self.cs2p[func][state]] = self.division_rates[func][state]
        return drm

    # fill the matrix of birth rates of cells in different states
    # (e.g. rates of a new cell in a given state appearing due to cell division)
    def fill_birth_matrix(self):
        # initialise the birth matrix
        brm=np.zeros((self.num_css,self.num_css))

        # fill the matrix
        for func_mother in self.cs2p.keys():
            for state in self.cs2p[func_mother].keys():
                # a cell with a given gene functionality divides into two cells with the same functionality
                for func_daughter in self.cs2p.keys():
                    chance_daughter_arising = 1 # start by assuming the possible daughter arises from the mother cell all the time
                    for  i in range(0,len(func_mother)):    # go through all genes to consider their mutations
                        if(func_mother[i]==0 and func_daughter[i]==1):
                            chance_daughter_arising *= 0    # if the supposed daughter has a gene that the mother doesn't, the daughter cannot arise
                        elif(func_mother[i]==1 and func_daughter[i]==0):
                            chance_daughter_arising *= self.mutation_rates[i] # chance of a given gene mutating
                        elif(func_mother[i]==1 and func_daughter[i]==1):
                            chance_daughter_arising *= 1-self.mutation_rates[i] # chance of a given gene remaining unmutated
                        # if both the mother and the daughter have a mutated gene copy, it always stays this way anyhow

                    # define the birth rate matrix entry as the product of the chance of the daughter arising and the growth rate
                    brm[self.cs2p[func_daughter][state],self.cs2p[func_mother][state]] = 2 * self.division_rates[func_mother][state] * chance_daughter_arising

        return brm

    # fill the matrix of switch state transition rates
    def fill_transition_matrix(self):
        # initialise the transition rate matrix
        trm=np.zeros((self.num_css,self.num_css))

        # fill the matrix
        for func in self.cs2p.keys():
            for state in self.cs2p[func].keys():
                # fill the matrix with the rates of transitions from the given state to all other states
                for state_to in self.cs2p[func].keys():
                    trm[self.cs2p[func][state_to],self.cs2p[func][state]]=self.transition_rates[func][state][state_to]
                # transitioning to other states means the current one is left
                trm[self.cs2p[func][state],self.cs2p[func][state]]=-np.sum(trm[:,self.cs2p[func][state]])

        return trm

    # fill the matrix of integrase action transitions
    def fill_integrase_matrix(self):
        # initialise the integrase action matrix
        irm=np.zeros((self.num_css,self.num_css))

        # fill the matrix
        for func in self.cs2p.keys():
            for state in self.cs2p[func].keys():
                # a cell with cut-out CAT arises due top the integrase
                func_to=func[0:3]+(0,) # integrase cuts out the CAT gene
                irm[self.cs2p[func_to][state],self.cs2p[func][state]] += \
                    self.integrase_activities[self.func2string[func]][state]

                # but there is no one fewer cell with CAT intact
                irm[self.cs2p[func][state],self.cs2p[func][state]] -= \
                    self.integrase_activities[self.func2string[func]][state]

        return irm

    # ANALYSING TRAJECTORIES -------------------------------------------------------------------------------------------
    # TOTAL abundance of all cells with the burden gene present
    def all_with_b(self, ts, xs):
        N_with_b = np.zeros_like(ts)
        for func in self.cs2p.keys():
            if (func[0] == 1):
                for state in self.cs2p[func].keys():
                    N_with_b += xs[:, self.cs2p[func][state]]
        return N_with_b

    # per-cell average synthesis rate of the burdensome protein - see Ingram and Stan 2023
    def percell_avg_synth_rate(self,
                               ts, xs
                               ):
        # get the syntesis rate unscaled by population size
        Hrates_unscaled = np.sum(np.multiply(np.multiply(self.pxv,self.drv),xs),axis=1)
        # for func in self.cs2p.keys():
        #     if(func[0]==1 and func[3]==1): # both burdensome and CAT genes present
        #         for state in self.cs2p[func].keys():
        #             Hrate_unscaled += self.pxv[self.cs2p[func][state]] * self.grv[self.cs2p[func][state]] * xs[:,self.cs2p[func][state]] / np.log2
        #     elif(func[0]==1 and func[3]==0): # just the burdensome gene present
        #         for state in self.cs2p[func].keys():
        #             Hrate_unscaled += self.pxv[self.cs2p[func][state]] * self.grv[self.cs2p[func][state]] * xs[:, self.cs2p[func][state]] / np.log2

        # scale by population size to enable comparisons between different populations
        Hrates = np.divide(Hrates_unscaled,np.sum(xs,axis=1))

        # return
        return Hrates

    # per-cell yield over the whole time
    def percell_yield(self,
                      ts, Hrates):

        return np.trapz(Hrates, ts)

    # function duration until synthesis rate falls below share_initial of the initial value
    def func_duration(self,
                        ts, Hrates,
                        share_initial=0.05):
        less_than_share_initial = np.where(Hrates < share_initial * Hrates[0])[0]
        if(len(less_than_share_initial)==0):
            return ts[-1]
        else:
            return ts[less_than_share_initial[0]]

    # PLOTTING ---------------------------------------------------------------------------------------------------------
    # plot the population breakdown by GENETIC state
    def plot_funcstates(self,
                        ts, xs,
                        dimensions=(640, 360),
                        tspan=None,
                        legend_label_style='code'):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create a ColumnDataSource object for the plot
        data={}
        for func in self.cs2p.keys():
            data[self.func2string[func]]=np.sum((xs[:,self.cs2p[func]['0']],
                                                    xs[:,self.cs2p[func]['L']],
                                                    xs[:,self.cs2p[func]['H']]
                                                ),axis=0)
        source = bkmodels.ColumnDataSource(data=data)

        # PLOT
        func_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="Time since start of culture, h",
            y_axis_label="Number of cells in genetic state",
            x_range=tspan,
            title='Number of cells',
            tools="box_zoom,pan,hover,reset,save"
        )
        palette = bkpalettes.Category20[len(self.cs2p.keys())]
        line_cntr = 0
        for func in self.cs2p.keys():
            func_figure.line(ts, data[self.func2string[func]],
                             legend_label=self.legend_labels[legend_label_style][func],
                             line_width=2, line_color=palette[line_cntr])
            line_cntr += 1

        # FORMAT LEGEND
        func_figure.legend.location = "right"
        func_figure.legend.click_policy = "hide"
        func_figure.legend.label_text_font_size = "6pt"
        func_figure.legend.background_fill_alpha = 1
        func_figure.legend.border_line_color = 'gray'
        func_figure.legend.margin = 2
        func_figure.legend.padding = 2
        func_figure.legend.spacing = 0

        return func_figure

    # plot the breakdown of a GENETIC state by SWITCH states
    def plot_funcstate_switches(self,
                                ts, xs,
                                func,
                                dimensions=(640, 360),
                                tspan=None,
                                legend_label_style='code'):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get the colour that corresponds to the genetic state
        palette = bkpalettes.Category20[len(self.cs2p)]
        func_colour = 0
        for func_possible in self.cs2p.keys():
            if(func_possible==func):
                break
            func_colour+=1

        # get the time points for plotting patches
        patch_ts=np.concatenate((ts,np.flip(ts)))

        # PLOT
        switch_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="Time, h",
            y_axis_label="Number of cells",
            x_range=tspan,
            title='Switch states of '+self.legend_labels[legend_label_style][func]+' cells',
            tools="box_zoom,pan,hover,reset"
        )
        # plot high state
        top_line = xs[:, self.cs2p[func]['H']] + xs[:, self.cs2p[func]['L']] + xs[:, self.cs2p[func]['0']]
        bottom_line = top_line - xs[:, self.cs2p[func]['H']]
        switch_figure.patch(patch_ts, np.concatenate((bottom_line, np.flip(top_line))),
                            fill_color=palette[func_colour], fill_alpha=0.5,
                            line_color=palette[func_colour],
                            legend_label='H', hatch_pattern='horizontal_wave')
        # plot low state
        top_line = bottom_line
        bottom_line = top_line - xs[:, self.cs2p[func]['L']]
        switch_figure.patch(patch_ts, np.concatenate((bottom_line, np.flip(top_line))),
                            fill_color=palette[func_colour], fill_alpha=0.5,
                            line_color=palette[func_colour],
                            legend_label='L', hatch_pattern='blank')
        # plot zero state
        top_line = bottom_line
        bottom_line = top_line - xs[:, self.cs2p[func]['0']]
        switch_figure.patch(patch_ts,np.concatenate((bottom_line,np.flip(top_line))),
                            fill_color=palette[func_colour], fill_alpha=0.5,
                            line_color=palette[func_colour],
                            legend_label='0', hatch_pattern='vertical_wave')

        # format legend
        switch_figure.legend.location = "top_right"
        switch_figure.legend.click_policy = "hide"
        switch_figure.legend.label_text_font_size = "6pt"

        return switch_figure

    # plot number of cells with burden gene present
    # (potentially comparing different systems' trajectories)
    def plot_with_burden(self,
                         ts1, xs1,
                         ts2=None, xs2=None,
                         label1=None, label2=None,
                         dimensions=(640, 360),
                         tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts1[0], ts1[-1])

        # get trajectories for plotting
        N_with_b1 = self.all_with_b(ts1, xs1)
        if(label2!=None):
            N_with_b2 = self.all_with_b(ts2, xs2)
        # PLOT
        burden_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="Time, h",
            y_axis_label="Number of cells",
            x_range=tspan,
            title='Cells with burdensome gene present',
            tools="box_zoom,pan,hover,reset,save"
        )
        # plot first trajectory
        burden_figure.line(ts1, N_with_b1,
                           legend_label=label1,
                           line_width=2, line_color='blue')
        # plot second trajectory if specified
        if(label2!=None):
            burden_figure.line(ts2, N_with_b2,
                               legend_label=label2,
                               line_width=2, line_color='red')
        # format legend
        burden_figure.legend.location = "top_right"
        burden_figure.legend.click_policy = "hide"
        burden_figure.legend.label_text_font_size = "6pt"

        return burden_figure

    # plot average protein synthesis rate of the burdensome protein per cell
    # (potentially comparing different systems' trajectories)
    def plot_percell_avg_synth_rate(self,
                         ts1, xs1,
                         ts2=None, xs2=None,
                         label1=None, label2=None,
                         dimensions=(640, 360),
                         tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts1[0], ts1[-1])

        # get trajectories for plotting
        Hrates1 = self.percell_avg_synth_rate(ts1, xs1)
        if (label2 != None):
            Hrates2 = self.percell_avg_synth_rate(ts2, xs2)
        # PLOT
        rate_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="Time, h",
            y_axis_label="Hrate, molecules/hour/cell",
            x_range=tspan,
            title='Average protein synthesis rate per cell',
            tools="box_zoom,pan,hover,reset,save"
        )
        # plot first trajectory
        rate_figure.line(ts1, Hrates1,
                           legend_label=label1,
                           line_width=2, line_color='blue')
        # plot second trajectory if specified
        if (label2 != None):
            rate_figure.line(ts2, Hrates2,
                               legend_label=label2,
                               line_width=2, line_color='red')
        # format legend
        rate_figure.legend.location = "top_right"
        rate_figure.legend.click_policy = "hide"
        rate_figure.legend.label_text_font_size = "6pt"

        return rate_figure

    # plot box plots of lifetime metrics
    def plot_metrics(self,
                     ts1, xs1,
                     share_initial=0.95,
                     ts2=None, xs2=None,
                     label1=None, label2=None,
                     dimensions=(640, 360),
                     tspan=None):
        # set x range
        if(label2!=None):
            x_range=[label1, label2]
        else:
            x_range=[label1]

        # get protein synthesis rates
        Hrates1 = self.percell_avg_synth_rate(ts1, xs1)
        if (label2 != None):
            Hrates2 = self.percell_avg_synth_rate(ts2, xs2)
        # get yields
        Hyield1 = self.percell_yield(ts1, Hrates1)
        if (label2 != None):
            Hyield2 = self.percell_yield(ts2, Hrates2)
        # get function durations
        funcdur1 = self.func_duration(ts1, Hrates1)
        if (label2 != None):
            funcdur2 = self.func_duration(ts2, Hrates2)

        # PLOT YIELDS
        yield_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_range=x_range,
            x_axis_label="Time, h",
            y_axis_label="Hyield, molecules/hour/cell",
            title='Average protein yield per cell over time',
            tools="box_zoom,pan,hover,reset,save"
        )
        # plot the box plot
        if (label2 != None):
            yield_figure.vbar(x=[label1, label2], width=0.75,
                              top=[Hyield1, Hyield2],
                              color=['blue', 'red'])
        else:
            yield_figure.vbar(x=[label1], width=0.75, top=[Hyield1],
                              color=['blue'])

        # PLOT FUNCTION DURATIONS
        funcdur_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_range=x_range,
            x_axis_label="t, h",
            y_axis_label="Function duration, h",
            title='Time taken by Hrate to decay to '+str(100*share_initial)+'% of its initial value',
            tools="box_zoom,pan,hover,reset,save"
        )
        # plot the box plot
        if (label2 != None):
            funcdur_figure.vbar(x=[label1, label2], width=0.75,
                              top=[funcdur1, funcdur2],
                              color=['blue', 'red'])
        else:
            funcdur_figure.vbar(x=[label1], width=0.75, top=[funcdur1],
                              color=['blue'])

        return yield_figure, funcdur_figure

    # dilution rate of the cell population
    def plot_dilution(self,
                      ts, xs,
                      dimensions=(640, 360),
                      tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get the dilution rate
        dilution_rate = np.divide(np.sum(np.multiply(self.drv,xs),axis=1),np.sum(xs,axis=1))

        # PLOT
        dilution_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="Time, h",
            y_axis_label="Dilution rate, 1/h",
            x_range=tspan,
            title='Dilution rate of the cell population',
            tools="box_zoom,pan,hover,reset,save"
        )
        dilution_figure.line(ts, dilution_rate,
                             line_width=2, line_color='blue')

        return dilution_figure

# ODE SIMULATION -------------------------------------------------------------------------------------------------------
# simulate the ODE using diffrax
def pop_ode_sim(
            # initial condition
            x0,
            # simulation parameters: time frame, when to save the system's state, relative and absolute tolerances
            tf, ts, rtol, atol,
            # population simulator
            pop_sim
            ):
    # set up jax

    # define the ODE term
    vector_field = lambda t, y, args: pop_ode(t, y, args)
    term = ODETerm(vector_field)

    # define arguments of the ODE term
    args = (
        jnp.array(pop_sim.drv), # division rates vector
        jnp.array(pop_sim.drm), # division rate matrix
        jnp.array(pop_sim.brm), # birth rate matrix
        jnp.array(pop_sim.trm), # transition rate matrix
        jnp.array(pop_sim.irm), # integrase rate matrix
    )

    # define the solver
    solver = Heun()

    # define the time points at which we save the solution
    stepsize_controller = PIDController(rtol=rtol, atol=atol)
    saveat = SaveAt(ts=ts)

    # solve the ODE
    sol = diffeqsolve(term, solver,
                      args=args,
                      t0=tf[0], t1=tf[1], dt0=0.1, y0=x0, saveat=saveat,
                      max_steps=None,
                      stepsize_controller=stepsize_controller)

    # convert jax arrays into numpy arrays
    return sol


# ode
@jax.jit
def pop_ode(t,x,args):
    # unpack arguments
    drv=args[0]
    drm=args[1]
    brm=args[2]
    trm=args[3]
    irm=args[4]

    dilution_rate=jnp.dot(x,drv)/np.sum(x) # find the dilution rate

    dxdt=(
        # cells divide
        - jnp.matmul(drm,x)
        # new cells are born due to cell division
        + jnp.matmul(brm,x)
        # cells transition from other states into the one in question
        + jnp.matmul(trm,x)
        # # integrase action: cell in one genetic state goes to another
        + jnp.matmul(irm,x)
        # account for dilution
        - dilution_rate*x
    )
    return dxdt


# CALCULATING INTEGRASE ACTIVITY FROM ITS CONCENTRATION AND CELL GROWTH RATE -------------------------------------------
def intact_calc(p_int, l, par, c_cat_pb):
    intact_rate=(p_int**4)/(p_int**4+par['K_bI~']**4) * par['k_sxf']/par['k_sxr'] * (par['k_+syn']+l) # integrase action rate
    c_cat_pb_int=int(c_cat_pb) # CAT gene copy number
    exp_time_to_all_cutouts=(1/(intact_rate)) * np.sum(1/np.arange(1,c_cat_pb_int+0.1,1)) # expected time to all cutouts - i.e. expected maximimum of c_cat exp. dist. samples
    return 1/exp_time_to_all_cutouts # integrase activity