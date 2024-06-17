'''
POP_SIMULATOR.PY: a class for simulating populations of cells with the punisher and a burdensome gene
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import functools
import itertools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
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
                 p_xtras=None, # burdensome protein concentrations
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

        # burdensome protein concentrations - in  a vector
        self.pxv = np.zeros(self.num_css)
        if(p_xtras!=None):
            for func in self.cs2p.keys():
                for state in self.cs2p[func].keys():
                    self.pxv[self.cs2p[func][state]] = p_xtras[self.func2string[func]][state]
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

    # FILLING MATRICES
    # all in format [to,from]
    # fill the matrix of cell division rates
    def fill_division_matrix(self):
        # initialise the division rate matrix
        drm=np.zeros((self.num_css,self.num_css))
        # just get a diagonal matrix with the growth rate of a given mutation-switch cell state
        for func in self.cs2p.keys():
            for state in self.cs2p[func].keys():
                drm[self.cs2p[func][state], self.cs2p[func][state]] = self.growth_rates[func][state]
        return drm

    # fill the matrix of birth rates of cells in different states
    # (e.g. rates of a new cell in a given state appearing due to cell division)
    def fill_birth_matrix(self):
        # initialise the birth matrix
        brm=np.zeros((self.num_css,self.num_css))

        # initialise the matrix of mutation probabilities
        mrm=np.zeros((self.num_css,self.num_css))

        # create a matrix of mutant progenies
        mpm_immediate={} # immediate progeny (obtainable by a single mutation)
        mpm={} # all progeny
        for func_parent in self.cs2p.keys():
            mpm_immediate[func_parent]=[]
            mpm[func_parent]=[]
            for func_child in self.cs2p.keys():
                # check if the child is an immediate progeny or just a progeny
                mutated_genes_in_child=0
                same_or_progeny=True
                for i in range(0,len(func_parent)):
                    if(func_parent[i]==1 and func_child[i]==0):
                        mutated_genes_in_child+=1
                    elif((func_parent[i]==0 and func_child[i]==1)):
                        same_or_progeny=False
                        break
                # make according entries in progeny lists
                if(same_or_progeny and not mutated_genes_in_child==0):
                    mpm[func_parent].append(func_child)
                    if(mutated_genes_in_child==1):
                        mpm_immediate[func_parent].append(func_child)

        # calculate the mutation probabilities for each genetic and switch state
        for func in self.cs2p.keys():
            for state in self.cs2p[func].keys():
                # travel down the matrix of mutation relations to find the ultimate child
                parents=[func]
                child_generation=0
                progeny_by_generation=[[]]
                while((0,0,0,0) not in parents):
                    child_generation+=1
                    progeny_by_generation.append([])
                    # find the next generation of children
                    for parent in parents:
                        for child in mpm_immediate[parent]:
                            if(child not in progeny_by_generation[child_generation]):
                                progeny_by_generation[child_generation].append(child)
                    # this generation's children will next be parents
                    parents=progeny_by_generation[child_generation]
                
                # calculate the probability of mutating into each child
                for generation in range(len(progeny_by_generation)-1,0,-1):
                    for descendant in progeny_by_generation[generation]:
                        # find probability of each mutation happening
                        mutation_prob=1
                        for i in range(0,len(func)):
                            if(func[i]==1 and descendant[i]==0):
                                mutation_prob*=self.mutation_rates[i]
                        mrm[self.cs2p[descendant][state],self.cs2p[func][state]]=mutation_prob
                        
                        # subtract the probability of MORE mutations happening (i.e. the child mutating further into an mpm member)
                        more_mutations_prob=0
                        for descendants_progeny in mpm[descendant]:
                            more_mutations_prob+=mutation_prob*mrm[self.cs2p[descendants_progeny][state],self.cs2p[descendant][state]]
                        mrm[self.cs2p[descendant][state],self.cs2p[func][state]]-=more_mutations_prob

                # according to mutation probabilities, calculate the birth rates
                # new mutant children being born
                for descendant in mpm[func]:
                    brm[self.cs2p[descendant][state],self.cs2p[func][state]]=2*self.growth_rates[func][state]*mrm[self.cs2p[descendant][state],self.cs2p[func][state]]
                # unmutated cell being born
                brm[self.cs2p[func][state],self.cs2p[func][state]]=2*self.growth_rates[func][state]*(1-np.sum(mrm[:,self.cs2p[func][state]]))

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
    def all_with_xtra(self, ts, xs):
        N_with_xtra = np.zeros_like(ts)
        for func in self.cs2p.keys():
            if (func[0] == 1):
                for state in self.cs2p[func].keys():
                    N_with_xtra += xs[:, self.cs2p[func][state]]
        return N_with_xtra

    # per-cell average synthesis rate of the burdensome protein - see Ingram and Stan 2023
    def percell_avg_synth_rate(self,
                               ts, xs
                               ):
        # get the syntesis rate unscaled by population size
        Hrates_unscaled = np.sum(np.multiply(np.multiply(self.pxv,self.grv),xs)/np.log(2),axis=1)
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
                        tspan=None):
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
            x_axis_label="Time, h",
            y_axis_label="Number of cells",
            x_range=tspan,
            title='Number of cells',
            tools="box_zoom,pan,hover,reset,save"
        )
        palette = bkpalettes.Category20[len(self.cs2p.keys())]
        line_cntr = 0
        for func in self.cs2p.keys():
            func_figure.line(ts, data[self.func2string[func]],
                             legend_label=self.func2string[func],
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
                                tspan=None):
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
            title='Switch states of '+self.func2string[func]+' cells',
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
        N_with_xtra1 = self.all_with_xtra(ts1, xs1)
        if(label2!=None):
            N_with_xtra2 = self.all_with_xtra(ts2, xs2)
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
        burden_figure.line(ts1, N_with_xtra1,
                           legend_label=label1,
                           line_width=2, line_color='blue')
        # plot second trajectory if specified
        if(label2!=None):
            burden_figure.line(ts2, N_with_xtra2,
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
        dilution_rate = np.divide(np.sum(np.multiply(self.grv,xs),axis=1),np.sum(xs,axis=1))

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
        jnp.array(pop_sim.grv), # growth rates vector
        jnp.array(pop_sim.drm), # division rate matrix
        jnp.array(pop_sim.brm), # birth rate matrix
        jnp.array(pop_sim.trm), # transition rate matrix
        jnp.array(pop_sim.irm), # integrase rate matrix
    )

    # define the solver
    solver = Kvaerno3()

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
    grv=args[0]
    drm=args[1]
    brm=args[2]
    trm=args[3]
    irm=args[4]

    dilution_rate=jnp.dot(x,grv)/np.sum(x) # find the dilution rate

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