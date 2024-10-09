'''
CELL_MODEL.PY: Python/Jax implementation of the coarse-grained resource-aware E.coli model
Class to enable resource-aware simulation of synthetic gene expression in the cell

Version where THE PROTEASE IS A SYNTHETIC PROTEIN WHOSE CONC. AFFECTS DEGRADATION RATES
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Kvaerno3, Heun, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts

import time

# CIRCUIT AND EXTERNAL INPUT IMPORTS -----------------------------------------------------------------------------------
import synthetic_circuits as circuits


# CELL MODEL FUNCTIONS -------------------------------------------------------------------------------------------------
# Definitions of functions appearing in the cell model ODEs
# apparent mRNA-ribosome dissociation constant
def k_calc(e, kplus, kminus, n):
    return (kminus + e / n) / kplus


# translation elongation rate
def e_calc(par, tc):
    return par['e_max'] * tc / (tc + par['K_e'])


# growth rate
def l_calc(par, e, B, prodeflux):
    return (e * B - prodeflux) / par['M']


# tRNA charging rate
def nu_calc(par, tu, s, metab_burd):
    return par['nu_max'] * s * (tu / (tu + par['K_nu'])) * (1 / (1 + metab_burd))


# tRNA synthesis rate
def psi_calc(par, T):
    return par['psi_max'] * T / (T + par['tau'])


# ribosomal gene transcription regulation function
def Fr_calc(par, T):
    return T / (T + par['tau'])


# CELL MODEL AUXILIARIES -----------------------------------------------------------------------------------------------
# Auxiliries for the cell model - set up default parameters and initial conditions, plot simulation outcomes
class CellModelAuxiliary:
    # INITIALISE
    def __init__(self):
        # plotting colours
        self.gene_colours = {'a': "#EDB120", 'r': "#7E2F8E", 'q': '#C0C0C0',
                             # colours for metabolic, riboozyme, housekeeping genes
                             'het': "#0072BD",
                             'h': "#77AC30"}  # colours for heterologous genes and intracellular chloramphenicol level
        self.tRNA_colours = {'tc': "#000000", 'tu': "#ABABAB"}  # colours for charged and uncharged tRNAs
        return

    # PROCESS SYNTHETIC CIRCUIT MODULE
    # add synthetic circuit to the cell model
    def add_circuit(self,
                    circuit_initialiser,  # function initialising the circuit
                    circuit_ode,  # function defining the circuit ODEs
                    circuit_F_calc,  # function calculating the circuit genes' transcription regulation functions
                    circuit_eff_m_het_div_k_het,
                    # function calculating the effective total mRNA/k for all synthetic genes
                    cellmodel_par, cellmodel_init_conds,  # host cell model parameters and initial conditions
                    # optional support for hybrid simulations
                    circuit_v=None,
                    varvol=False  # whether the hybrid simulation considers variable cell volumes
                    ):
        # call circuit initialiser
        circuit_par, circuit_init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles = circuit_initialiser()

        # update parameter, initial condition and colour dictionaries
        cellmodel_par.update(circuit_par)
        cellmodel_init_conds.update(circuit_init_conds)

        # join the circuit ODEs with the transcription regulation functions
        circuit_ode_with_F_calc = lambda t, x, e, l, R, k_het, D, p_prot, par, name2pos: circuit_ode(circuit_F_calc,
                                                                                                     t, x, e, l, R,
                                                                                                     k_het, D, p_prot,
                                                                                                     par, name2pos)

        # IF stochastic component specified, predefine F_calc for it as well
        if (circuit_v != None):
            if not varvol:
                circuit_v_with_F_calc = lambda t, x, e, l, R, k_het, D, p_prot, mRNA_count_scales, par, name2pos, : circuit_v(
                    circuit_F_calc,
                    t, x, e, l, R, k_het, D, p_prot,
                    mRNA_count_scales,
                    par, name2pos)
            else:
                circuit_v_with_F_calc = lambda t, x, e, l, R, k_het, D, p_prot, mRNA_count_scales, par, name2pos, \
                                               V, rep_vols: circuit_v(
                    circuit_F_calc,
                    t, x, e, l, R, k_het, D, p_prot,
                    mRNA_count_scales,
                    par, name2pos, V, rep_vols)
        else:
            circuit_v_with_F_calc = None

        # add the ciorcuit ODEs to that of the host cell model
        cellmodel_ode = lambda t, x, args: ode(t, x, circuit_ode_with_F_calc, circuit_eff_m_het_div_k_het, args)

        # return updated ode and parameter, initial conditions, circuit gene (and miscellaneous specie) names
        # name - position in state vector decoder and colours for plotting the circuit's time evolution
        return cellmodel_ode, circuit_F_calc, circuit_eff_m_het_div_k_het, \
            cellmodel_par, cellmodel_init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles, circuit_v_with_F_calc

    # package synthetic gene parameters into jax arrays for calculating k values
    def synth_gene_params_for_jax(self, par,  # system parameters
                                  circuit_genes  # circuit gene names
                                  ):
        # initialise parameter arrays
        kplus_het = np.zeros(len(circuit_genes))
        kminus_het = np.zeros(len(circuit_genes))
        n_het = np.zeros(len(circuit_genes))
        d_het = np.zeros(len(circuit_genes))
        g_het = np.zeros(len(circuit_genes))

        # fill parameter arrays
        for i in range(0, len(circuit_genes)):
            kplus_het[i] = par['k+_' + circuit_genes[i]]
            kminus_het[i] = par['k-_' + circuit_genes[i]]
            n_het[i] = par['n_' + circuit_genes[i]]
            d_het[i] = par['d_' + circuit_genes[i]]
            g_het[i] = par['g_' + circuit_genes[i]]

        # return as a tuple of arrays
        return (jnp.array(kplus_het), jnp.array(kminus_het), jnp.array(n_het), jnp.array(d_het), jnp.array(g_het))

    # SET DEFAULTS
    # set default parameters
    def default_params(self):
        '''
        Chloramphenicol-related parameters taken from:
        Gutiérrez Mena J et al. 2022 Dynamic cybergenetic control of bacterial co-culture composition via optogenetic feedback
        All other parameters taken from:
        Sechkar A et al. 2024 A coarse-grained bacterial cell model for resource-aware analysis and design of synthetic gene circuits
        '''

        params = {}  # initialise

        # GENERAL PARAMETERS
        params['M'] = 1.19e9  # cell mass (aa)
        params['phi_q'] = 0.59  # constant housekeeping protein mass fraction

        # GENE EXPRESSION parAMETERS
        # metabolic/aminoacylating genes
        params['c_a'] = 1.0  # copy no. (nM) - convention
        params['b_a'] = 6.0  # mRNA decay rate (/h)
        params['k+_a'] = 60.0  # ribosome binding rate (/h/nM)
        params['k-_a'] = 60.0  # ribosome unbinding rate (/h)
        params['n_a'] = 300.0  # protein length (aa)

        # ribosomal gene
        params['c_r'] = 1.0  # copy no. (nM) - convention
        params['b_r'] = 6.0  # mRNA decay rate (/h)
        params['k+_r'] = 60.0  # ribosome binding rate (/h/nM)
        params['k-_r'] = 60.0  # ribosome unbinding rate (/h)
        params['n_r'] = 7459.0  # protein length (aa)

        # ACTIVATION & RATE FUNCTION PARAMETERS
        params['e_max'] = 7.2e4  # max translation elongation rate (aa/h)
        params['psi_max'] = 4.32e5  # max tRNA syntheis rate per untig rowth rate (nM)
        params['tau'] = 1.0  # ppGpp sensitivity (ribosome transc. and tRNA synth. Hill const)

        # CHLORAMPHENICOL-RELATED PARAMETERS
        params['h_ext'] = 0.0  # chloramphenicol concentration in the culture medium (nM)
        params['diff_h'] = 90.0 * 60  # chloramphenicol diffusion coefficient through the cell membrane (1/h)
        params['K_D'] = 1300.0  # chloramphenicol-ribosome dissociation constant (nM)
        params['K_C'] = (
                                1 / 3) / 60  # dissociation constant for chloramphenicol removal by the cat (chloramphenicol resistance) protein, if present (nM*h)
        params[
            'cat_gene_present'] = 0  # 1 if cat gene is present, 0 otherwise (will be automatically set to 1 if your circuit has a gene titled 'cat' and you haven't messed where you shouldn't)
        params['eff_h'] = 0.0  # chloramphenicol efflux rate due to membrane protein activity (1/h)

        # PARAMETERS FITTED TO DATA IN SECHKAR ET AL., 2024
        params['a_a'] = 394464.6979  # metabolic gene transcription rate (/h)
        params['a_r'] = 1.0318 * params['a_a']  # ribosomal gene transcription rate (/h)
        params['nu_max'] = 4.0469e3  # max tRNA amioacylation rate (/h)
        params['K_nu'] = 1.2397e3  # tRNA charging rate Michaelis-Menten constant (nM)
        params['K_e'] = 1.2397e3  # translation elongation rate Michaelis-Menten constant (nM)
        return params

    # set default initial conditions
    def default_init_conds(self, par):
        init_conds = {}  # initialise

        # mRNA concentrations - non-zero to avoid being stuck at lambda=0
        init_conds['m_a'] = 1000.0  # metabolic
        init_conds['m_r'] = 0.01  # ribosomal

        # protein concentrations - start with 50/50 a/R allocation as a convention
        init_conds['p_a'] = par['M'] * (1 - par['phi_q']) / (2 * par['n_a'])  # metabolic *
        init_conds['R'] = par['M'] * (1 - par['phi_q']) / (2 * par['n_r'])  # ribosomal *

        # tRNA concentrations - 3E-5 abundance units in Chure and Cremer 2022 are equivalent to 80 uM = 80000 nM
        init_conds['tc'] = 80000.0  # charged tRNAs
        init_conds['tu'] = 80000.0  # free tRNAs

        # nutrient quality s and chloramphenicol concentration h
        init_conds['s'] = 0.5
        init_conds['h'] = 0.0  # no translation inhibition assumed by default
        return init_conds

    # PREPARE FOR SIMULATIONS
    # set default initial condition vector
    def x0_from_init_conds(self, init_conds, circuit_genes, circuit_miscs):
        # NATIVE GENES
        x0 = [
            # mRNAs;
            init_conds['m_a'],  # metabolic gene transcripts
            init_conds['m_r'],  # ribosomal gene transcripts

            # proteins
            init_conds['p_a'],  # metabolic proteins
            init_conds['R'],  # non-inactivated ribosomes

            # tRNAs
            init_conds['tc'],  # charged
            init_conds['tu'],  # uncharged

            # culture medium's nutrient quality and chloramphenicol concentration
            init_conds['s'],  # nutrient quality
            init_conds['h'],  # chloramphenicol levels IN THE CELL
        ]
        # SYNTHETIC CIRCUIT
        for gene in circuit_genes:  # mRNAs
            x0.append(init_conds['m_' + gene])
        for gene in circuit_genes:  # proteins
            x0.append(init_conds['p_' + gene])

        # MISCELLANEOUS SPECIES
        for misc in circuit_miscs:  # miscellanous species
            x0.append(init_conds[misc])

        return jnp.array(x0)

    # PLOT RESULTS, CALCULATE CELLULAR VARIABLES
    # plot protein composition of the cell by mass over time
    def plot_protein_masses(self, ts, xs,
                            par, circuit_genes,  # model parameters, list of circuit genes
                            dimensions=(320, 180), tspan=None,
                            varvol=False  # whether the simulation considers variable cell volumes
                            ):
        if (varvol):
            Vs = xs[:, 6]  # cell volumes
            xs_concs = np.divide(xs, Vs * np.ones_like(
                np.array([xs[0, :]]).T))  # divide abundances by cell volumes to get concentrations
            xs_concs[:, 6] = par['s'] * np.ones_like(
                Vs)  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xs_concs = xs

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # create figure
        mass_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Protein mass, aa",
            x_range=tspan,
            title='Protein masses',
            tools="box_zoom,pan,hover,reset"
        )

        flip_t = np.flip(ts)  # flipped time axis for patch plotting

        # plot heterologous protein mass - if there are any heterologous proteins to begin with
        if (len(circuit_genes) != 0):
            bottom_line = np.zeros(xs_concs.shape[0])
            top_line = bottom_line + np.sum(xs_concs[:, 8 + len(circuit_genes):8 + len(circuit_genes) * 2] * np.array(
                self.synth_gene_params_for_jax(par, circuit_genes)[2], ndmin=2), axis=1)
            mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                              line_width=0.5, line_color='black', fill_color=self.gene_colours['het'],
                              legend_label='het')
        else:
            top_line = np.zeros(xs_concs.shape[0])

        # plot mass of inactivated ribosomes
        if ((xs_concs[:, 7] != 0).any()):
            bottom_line = top_line
            top_line = bottom_line + xs_concs[:, 3] * par['n_r'] * (xs_concs[:, 7] / (par['K_D'] + xs_concs[:, 7]))
            mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                              line_width=0.5, line_color='black', fill_color=self.gene_colours['h'], legend_label='R:h')

        # plot mass of active ribosomes - only if there are any to begin with
        bottom_line = top_line
        top_line = bottom_line + xs_concs[:, 3] * par['n_r'] * (par['K_D'] / (par['K_D'] + xs_concs[:, 7]))
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['r'],
                          legend_label='R (free)')

        # plot metabolic protein mass
        bottom_line = top_line
        top_line = bottom_line + xs_concs[:, 2] * par['n_a']
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['a'], legend_label='p_a')

        # plot housekeeping protein mass
        bottom_line = top_line
        top_line = bottom_line / (1 - par['phi_q'])
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['q'], legend_label='p_q')

        # add legend
        mass_figure.legend.label_text_font_size = "8pt"
        mass_figure.legend.location = "top_right"

        return mass_figure

    # plot mRNA, protein and tRNA concentrations over time
    def plot_native_concentrations(self, ts, xs,
                                   par, circuit_genes,  # model parameters, list of circuit genes
                                   dimensions=(320, 180), tspan=None,
                                   varvol=False  # whether the simulation considers variable cell volumes
                                   ):
        if (varvol):
            Vs = xs[:, 6]  # cell volumes
            xs_concs = np.divide(xs, (Vs * np.ones_like(
                np.array([xs[0, :]]).T)).T)  # divide abundances by cell volumes to get concentrations
            xs_concs[:, 6] = par['s'] * np.ones_like(
                Vs)  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xs_concs = xs

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create a ColumnDataSource object for the plot
        source = bkmodels.ColumnDataSource(data={
            't': ts,
            'm_a': xs_concs[:, 0],  # metabolic mRNA
            'm_r': xs_concs[:, 1],  # ribosomal mRNA
            'p_a': xs_concs[:, 2],  # metabolic protein
            'R': xs_concs[:, 3],  # ribosomal protein
            'tc': xs_concs[:, 4],  # charged tRNA
            'tu': xs_concs[:, 5],  # uncharged tRNA
            's': xs_concs[:, 6],  # nutrient quality
            'h': xs_concs[:, 7],  # chloramphenicol concentration
            'm_het': np.sum(xs_concs[:, 8:8 + len(circuit_genes)], axis=1),  # heterologous mRNA
            'p_het': np.sum(xs_concs[:, 8 + len(circuit_genes):8 + len(circuit_genes) * 2], axis=1),
            # heterologous protein
        })

        # PLOT mRNA CONCENTRATIONS
        mRNA_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="mRNA conc., nM",
            x_range=tspan,
            title='mRNA concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        mRNA_figure.line(x='t', y='m_a', source=source, line_width=1.5, line_color=self.gene_colours['a'],
                         legend_label='m_a')  # plot metabolic mRNA concentrations
        mRNA_figure.line(x='t', y='m_r', source=source, line_width=1.5, line_color=self.gene_colours['r'],
                         legend_label='m_r')  # plot ribosomal mRNA concentrations
        mRNA_figure.line(x='t', y='m_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
                         legend_label='m_het')  # plot heterologous mRNA concentrations
        mRNA_figure.legend.label_text_font_size = "8pt"
        mRNA_figure.legend.location = "top_right"
        mRNA_figure.legend.click_policy = 'hide'

        # PLOT protein CONCENTRATIONS
        protein_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Protein conc., nM",
            x_range=tspan,
            title='Protein concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        protein_figure.line(x='t', y='p_a', source=source, line_width=1.5, line_color=self.gene_colours['a'],
                            legend_label='p_a')  # plot metabolic protein concentrations
        protein_figure.line(x='t', y='R', source=source, line_width=1.5, line_color=self.gene_colours['r'],
                            legend_label='R')  # plot ribosomal protein concentrations
        protein_figure.line(x='t', y='p_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
                            legend_label='p_het')  # plot heterologous protein concentrations
        protein_figure.legend.label_text_font_size = "8pt"
        protein_figure.legend.location = "top_right"
        protein_figure.legend.click_policy = 'hide'

        # PLOT tRNA CONCENTRATIONS
        tRNA_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA conc., nM",
            x_range=tspan,
            title='tRNA concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        tRNA_figure.line(x='t', y='tc', source=source, line_width=1.5, line_color=self.tRNA_colours['tc'],
                         legend_label='tc')  # plot charged tRNA concentrations
        tRNA_figure.line(x='t', y='tu', source=source, line_width=1.5, line_color=self.tRNA_colours['tu'],
                         legend_label='tu')  # plot uncharged tRNA concentrations
        tRNA_figure.legend.label_text_font_size = "8pt"
        tRNA_figure.legend.location = "top_right"
        tRNA_figure.legend.click_policy = 'hide'

        # PLOT INTRACELLULAR CHLORAMPHENICOL CONCENTRATION
        h_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="h, nM",
            x_range=tspan,
            title='Intracellular chloramphenicol concentration',
            tools="box_zoom,pan,hover,reset"
        )
        h_figure.line(x='t', y='h', source=source, line_width=1.5, line_color=self.gene_colours['h'],
                      legend_label='h')  # plot intracellular chloramphenicol concentration

        return mRNA_figure, protein_figure, tRNA_figure, h_figure

    # plot concentrations for the synthetic circuits
    def plot_circuit_concentrations(self, ts, xs,
                                    par, circuit_genes, circuit_miscs, circuit_name2pos,
                                    # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                    circuit_styles,  # colours for the circuit plots
                                    dimensions=(320, 180), tspan=None,
                                    varvol=False  # whether the simulation considers variable cell volumes
                                    ):
        # turn molecule counts into concentrations if considering variable cell volumes
        if (varvol):
            Vs = xs[:, 6]  # cell volumes
            xs_concs = np.divide(xs, (Vs * np.ones_like(
                np.array([xs[0, :]]).T)).T)  # divide abundances by cell volumes to get concentrations
            xs_concs[:, 6] = par['s'] * np.ones_like(
                Vs)  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xs_concs = xs

        # if no circuitry at all, return no plots
        if (len(circuit_genes) + len(circuit_miscs) == 0):
            return None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create a ColumnDataSource object for the plot
        data_for_column = {'t': ts}  # initialise with time axis
        # record synthetic mRNA and protein concentrations
        for i in range(0, len(circuit_genes)):
            data_for_column['m_' + circuit_genes[i]] = xs_concs[:, 8 + i]
            data_for_column['p_' + circuit_genes[i]] = xs_concs[:, 8 + len(circuit_genes) + i]
        # record miscellaneous species' concentrations
        for i in range(0, len(circuit_miscs)):
            data_for_column[circuit_miscs[i]] = xs_concs[:, 8 + len(circuit_genes) * 2 + i]
        source = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT mRNA and PROTEIN CONCENTRATIONS (IF ANY)
        if (len(circuit_genes) > 0):
            # mRNAs
            mRNA_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="mRNA conc., nM",
                x_range=tspan,
                title='mRNA concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            for gene in circuit_genes:
                mRNA_figure.line(x='t', y='m_' + gene, source=source, line_width=1.5,
                                 line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                                 legend_label='m_' + gene)
            mRNA_figure.legend.label_text_font_size = "8pt"
            mRNA_figure.legend.location = "top_right"
            mRNA_figure.legend.click_policy = 'hide'

            # proteins
            protein_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Protein conc., nM",
                x_range=tspan,
                title='Protein concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            for gene in circuit_genes:
                protein_figure.line(x='t', y='p_' + gene, source=source, line_width=1.5,
                                    line_color=circuit_styles['colours'][gene],
                                    line_dash=circuit_styles['dashes'][gene],
                                    legend_label='p_' + gene)
            protein_figure.legend.label_text_font_size = "8pt"
            protein_figure.legend.location = "top_right"
            protein_figure.legend.click_policy = 'hide'
        else:
            mRNA_figure = None
            protein_figure = None

        # PLOT MISCELLANEOUS SPECIES' CONCENTRATIONS (IF ANY)
        if (len(circuit_miscs) > 0):
            misc_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Conc., nM",
                x_range=tspan,
                title='Miscellaneous species concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            for misc in circuit_miscs:
                misc_figure.line(x='t', y=misc, source=source, line_width=1.5,
                                 line_color=circuit_styles['colours'][misc], line_dash=circuit_styles['dashes'][misc],
                                 legend_label=misc)
            misc_figure.legend.label_text_font_size = "8pt"
            misc_figure.legend.location = "top_right"
            misc_figure.legend.click_policy = 'hide'
        else:
            misc_figure = None

        return mRNA_figure, protein_figure, misc_figure

    # plot transcription regulation function values for the circuit's genes
    def plot_circuit_regulation(self, ts, xs,
                                circuit_F_calc,
                                # function calculating the transcription regulation functions for the circuit
                                par, circuit_genes, circuit_miscs, circuit_name2pos,
                                # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                circuit_styles,  # colours for the circuit plots
                                dimensions=(320, 180), tspan=None,
                                varvol=False  # whether the simulation considers variable cell volumes
                                ):
        # if considering variable cell volumes, divide molecule counts by cell volume to get concentrations
        if (varvol):
            Vs = xs[:, 6]  # cell volumes
            xs_concs = np.divide(xs, (Vs * np.ones_like(
                np.array([xs[0, :]]).T)).T)  # divide abundances by cell volumes to get concentrations
            xs_concs[:, 6] = par['s'] * np.ones_like(
                Vs)  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xs_concs = xs

        # if no circuitry, return no plots
        if (len(circuit_genes) == 0):
            return None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # find values of gene transcription regulation functions
        Fs = np.zeros((len(ts), len(circuit_genes)))  # initialise
        for i in range(0, len(ts)):
            Fs[i, :] = np.array(circuit_F_calc(ts[i], xs_concs[i, :], par, circuit_name2pos)[:])

        # Create ColumnDataSource object for the plot
        data_for_column = {'t': ts}  # initialise with time axis
        for i in range(0, len(circuit_genes)):
            data_for_column['F_' + str(circuit_genes[i])] = Fs[:, i]

        # PLOT
        F_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Transc. reg. funcs. F",
            x_range=tspan,
            y_range=(0, 1.05),
            title='Gene transcription regulation',
            tools="box_zoom,pan,hover,reset"
        )
        for gene in circuit_genes:
            F_figure.line(x='t', y='F_' + gene, source=data_for_column, line_width=1.5,
                          line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                          legend_label='F_' + gene)
        F_figure.legend.label_text_font_size = "8pt"
        F_figure.legend.location = "top_right"
        F_figure.legend.click_policy = 'hide'

        return F_figure

    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables(self, ts, xs,
                            par, circuit_genes, circuit_miscs, circuit_name2pos,
                            circuit_eff_m_het_div_k_het,
                            # function calculating the effective total mRNA/k for all synthetic genes
                            dimensions=(320, 180), tspan=None,
                            varvol=False  # whether the simulation considers variable cell volumes
                            ):
        # if considering variable cell volumes, divide molecule counts by cell volume to get concentrations
        if (varvol):
            Vs = xs[:, 6]  # cell volumes
            xs_concs = np.divide(xs, (Vs * np.ones_like(
                np.array([xs[0, :]]).T)).T)  # divide abundances by cell volumes to get concentrations
            xs_concs[:, 6] = par['s'] * np.ones_like(
                Vs)  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xs_concs = xs

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time
        e, l, F_r, nu, _, T, D, D_nodeg = self.get_e_l_Fr_nu_psi_T_D_Dnodeg(ts, xs_concs, par, circuit_genes,
                                                                            circuit_miscs,
                                                                            circuit_name2pos,
                                                                            circuit_eff_m_het_div_k_het)

        # Create a ColumnDataSource object for the plot
        data_for_column = {'t': np.array(ts), 'e': np.array(e), 'l': np.array(l), 'F_r': np.array(F_r),
                           'ppGpp': np.array(1 / T), 'nu': np.array(nu), 'D': np.array(D), 'D_nodeg': np.array(D_nodeg)}
        source = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT GROWTH RATE
        l_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Growth rate, 1/h",
            x_range=tspan,
            y_range=(0, 2),
            title='Growth rate',
            tools="box_zoom,pan,hover,reset"
        )
        l_figure.line(x='t', y='l', source=source, line_width=1.5, line_color='blue', legend_label='l')
        l_figure.legend.label_text_font_size = "8pt"
        l_figure.legend.location = "top_right"
        l_figure.legend.click_policy = 'hide'

        # PLOT TRANSLATION ELONGATION RATE
        e_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Translation elongation rate, 1/h",
            x_range=tspan,
            y_range=(0, par['e_max']),
            title='Translation elongation rate',
            tools="box_zoom,pan,hover,reset"
        )
        e_figure.line(x='t', y='e', source=source, line_width=1.5, line_color='blue', legend_label='e')
        e_figure.legend.label_text_font_size = "8pt"
        e_figure.legend.location = "top_right"
        e_figure.legend.click_policy = 'hide'

        # PLOT RIBOSOMAL GENE TRANSCRIPTION REGULATION FUNCTION
        Fr_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Ribosomal gene transcription regulation function",
            x_range=tspan,
            y_range=(0, 1),
            title='Ribosomal gene transcription regulation function',
            tools="box_zoom,pan,hover,reset"
        )
        Fr_figure.line(x='t', y='F_r', source=source, line_width=1.5, line_color='blue', legend_label='F_r')
        Fr_figure.legend.label_text_font_size = "8pt"
        Fr_figure.legend.location = "top_right"
        Fr_figure.legend.click_policy = 'hide'

        # PLOT ppGpp CONCENTRATION
        ppGpp_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Rel. ppGpp conc. = 1/T",
            x_range=tspan,
            title='ppGpp concentration',
            tools="box_zoom,pan,hover,reset"
        )
        ppGpp_figure.line(x='t', y='ppGpp', source=source, line_width=1.5, line_color='blue', legend_label='ppGpp')
        ppGpp_figure.legend.label_text_font_size = "8pt"
        ppGpp_figure.legend.location = "top_right"
        ppGpp_figure.legend.click_policy = 'hide'

        # PLOT tRNA CHARGING RATE
        nu_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA charging rate, aa/h",
            x_range=tspan,
            title='tRNA charging rate',
            tools="box_zoom,pan,hover,reset"
        )
        nu_figure.line(x='t', y='nu', source=source, line_width=1.5, line_color='blue', legend_label='nu')
        nu_figure.legend.label_text_font_size = "8pt"
        nu_figure.legend.location = "top_right"
        nu_figure.legend.click_policy = 'hide'

        # PLOT RC DENOMINATOR
        D_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="RC denominator D",
            x_range=tspan,
            title='Resource Competition denominator',
            tools="box_zoom,pan,hover,reset"
        )
        D_figure.line(x='t', y='D', source=source, line_width=1.5, line_color='blue', legend_label='D')
        # D_figure.line(x='t', y='D_nodeg', source=source, line_width=1.5, line_color='red', legend_label='D (no deg.)')
        D_figure.legend.label_text_font_size = "8pt"
        D_figure.legend.location = "top_right"
        D_figure.legend.click_policy = 'hide'

        return l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure

    # plot cell volume over time
    def plot_volume(self, ts, xs,
                    par, circuit_genes,  # model parameters, list of circuit genes
                    dimensions=(320, 180), tspan=None,  # whether the simulation considers variable cell volumes
                    ):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create a ColumnDataSource object for the plot
        source = bkmodels.ColumnDataSource(data={
            't': ts,
            'V': xs[:, 6],  # cel volume
        })

        vol_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Cell volume, um^3",
            x_range=tspan,
            title='Cell volume',
            tools="box_zoom,pan,hover,reset"
        )
        vol_figure.line(x='t', y='V', source=source, line_width=1.5, line_color='blue',
                        legend_label='cell volume')  # plot metabolic mRNA concentrations
        vol_figure.legend.label_text_font_size = "8pt"
        vol_figure.legend.location = "top_right"
        vol_figure.legend.click_policy = 'hide'

        return vol_figure

    # find values of different cellular variables
    def get_e_l_Fr_nu_psi_T_D_Dnodeg(self, t, x,
                                     par, circuit_genes, circuit_miscs, circuit_name2pos,
                                     circuit_eff_m_het_div_k_het):
        # give the state vector entries meaningful names
        m_a = x[:, 0]  # metabolic gene mRNA
        m_r = x[:, 1]  # ribosomal gene mRNA
        p_a = x[:, 2]  # metabolic proteins
        R = x[:, 3]  # non-inactivated ribosomes
        tc = x[:, 4]  # charged tRNAs
        tu = x[:, 5]  # uncharged tRNAs
        s = x[:, 6]  # nutrient quality (constant)
        h = x[:, 7]  # chloramphenicol concentration (constant)
        x_het = x[:, 8:8 + 2 * len(circuit_genes)]  # heterologous protein concentrations
        misc = x[:, 8 + 2 * len(circuit_genes):8 + 2 * len(circuit_genes) + len(circuit_miscs)]  # miscellaneous species

        # vector of Synthetic Gene Parameters 4 JAX
        sgp4j = self.synth_gene_params_for_jax(par, circuit_genes)
        kplus_het, kminus_het, n_het, d_het, g_het = sgp4j

        # FIND SPECIAL SYNTHETIC PROTEIN CONCENTRATIONS - IF PRESENT
        # chloramphenicol acetyltransferase (antibiotic reistance)
        p_cat = jax.lax.select(par['cat_gene_present'] == 1, x[:, circuit_name2pos['p_cat']], jnp.zeros_like(x[:, 0]))
        # synthetic protease (synthetic protein degradation)
        p_prot = jax.lax.select(par['prot_gene_present'] == 1, x[:, circuit_name2pos['p_prot']],
                                jnp.zeros_like(x[:, 0]))

        # CALCULATE PHYSIOLOGICAL VARIABLES
        # translation elongation rate
        e = e_calc(par, tc)

        # ribosome dissociation constants
        k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])
        k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])
        k_het = k_calc((jnp.atleast_2d(jnp.array(e) * jnp.ones((len(circuit_genes), 1)))).T,
                       jnp.atleast_2d(kplus_het) * jnp.ones((len(e), 1)),
                       jnp.atleast_2d(kminus_het) * jnp.ones((len(e), 1)),
                       jnp.atleast_2d(n_het) * jnp.ones((len(e), 1)))  # heterologous genes

        # ratio of charged to uncharged tRNAs
        T = tc / tu

        # corection to ribosome availability due to chloramphenicol action
        H = (par['K_D'] + h) / par['K_D']

        # heterologous mRNA levels scaled by RBS strength
        m_het_div_k_het_np = np.zeros(t.shape)
        for i in range(0, len(t)):
            m_het_div_k_het_np[i] = circuit_eff_m_het_div_k_het(x[i, :], par, circuit_name2pos, len(circuit_genes),
                                                                e[i], k_het[i, :])
        m_het_div_k_het = jnp.array(m_het_div_k_het_np)

        # heterologous protein degradation flux
        prodeflux = jnp.multiply(
            p_prot,
            jnp.sum(d_het * n_het * x[:, 8 + len(circuit_genes):8 + len(circuit_genes) * 2], axis=1)
        )
        # heterologous protein degradation flux
        prodeflux_times_H_div_eR = prodeflux * H / (e * R)

        # resource competition denominator
        m_notq_div_k_notq = m_a / k_a + m_r / k_r + m_het_div_k_het
        mq_div_kq = (par['phi_q'] * (1 - prodeflux_times_H_div_eR) * m_notq_div_k_notq - par[
            'phi_q'] * prodeflux_times_H_div_eR) / \
                    (1 - par['phi_q'] * (1 - prodeflux_times_H_div_eR))
        D = H * (1 + mq_div_kq + m_notq_div_k_notq)
        B = R * (1 / H - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

        # metabolic burdne
        metab_burd = jnp.sum(g_het * x[:, 8 + len(circuit_genes):8 + len(circuit_genes) * 2], axis=1)

        nu = nu_calc(par, tu, s, metab_burd)  # tRNA charging rate

        l = l_calc(par, e, B, prodeflux)  # growth rate

        psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

        F_r = Fr_calc(par, T)  # ribosomal gene transcription regulation function

        # RC denominator, as it would be without active protein degradation by the protease
        D_nodeg = H * (1 + (1 / (1 - par['phi_q'])) * m_notq_div_k_notq)
        return e, l, F_r, nu, jnp.multiply(psi, l), T, D, D_nodeg

    # PLOT RESULTS FOR SEVERAL TRAJECTORIES AT ONCE (SAME TIME AXIS)
    # plot mRNA, protein and tRNA concentrations over time
    def plot_native_concentrations_multiple(self, ts, xss,
                                            par, circuit_genes,  # model parameters, list of circuit genes
                                            dimensions=(320, 180), tspan=None,
                                            simtraj_alpha=0.1,
                                            varvol=False  # whether the simulation considers variable cell volumes
                                            ):
        # if considering variable cell volumes, divide heterologous molecule counts by cell volume
        if varvol:
            xss_concs = np.zeros_like(xss)  # initialise
            Vs = np.zeros_like(xss[:, :, 6])  # initialise
            for i in range(0, len(xss)):
                Vs[i, :] = xss[i, :, 6]  # cell volumes
                xss_concs[i, :, :] = np.divide(xss, (Vs[i, :] * np.ones_like(
                    np.array([xss[i, 0, :]]).T)).T)  # divide abundances by cell volumes to get concentrations
                xss_concs[i, :, 6] = par['s'] * np.ones_like(
                    Vs[i, :])  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xss_concs = xss

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss_concs)):
            # Create a ColumnDataSource object for the plot
            sources[i] = bkmodels.ColumnDataSource(data={
                't': ts,
                'm_a': xss_concs[i, :, 0],  # metabolic mRNA
                'm_r': xss_concs[i, :, 1],  # ribosomal mRNA
                'p_a': xss_concs[i, :, 2],  # metabolic protein
                'R': xss_concs[i, :, 3],  # ribosomal protein
                'tc': xss_concs[i, :, 4],  # charged tRNA
                'tu': xss_concs[i, :, 5],  # uncharged tRNA
                's': xss_concs[i, :, 6],  # nutrient quality
                'h': xss_concs[i, :, 7],  # chloramphenicol concentration
                'm_het': np.sum(xss_concs[i, :, 8:8 + len(circuit_genes)], axis=1),  # heterologous mRNA
                'p_het': np.sum(xss_concs[i, :, 8 + len(circuit_genes):8 + len(circuit_genes) * 2], axis=1),
                # heterologous protein
            })

        # Create a ColumnDataSource object for plotting the average trajectory
        source_avg = bkmodels.ColumnDataSource(data={
            't': ts,
            'm_a': np.mean(xss_concs[:, :, 0], axis=0),  # metabolic mRNA
            'm_r': np.mean(xss_concs[:, :, 1], axis=0),  # ribosomal mRNA
            'p_a': np.mean(xss_concs[:, :, 2], axis=0),  # metabolic protein
            'R': np.mean(xss_concs[:, :, 3], axis=0),  # ribosomal protein
            'tc': np.mean(xss_concs[:, :, 4], axis=0),  # charged tRNA
            'tu': np.mean(xss_concs[:, :, 5], axis=0),  # uncharged tRNA
            's': np.mean(xss_concs[:, :, 6], axis=0),  # nutrient quality
            'h': np.mean(xss_concs[:, :, 7], axis=0),  # chloramphenicol concentration
            'm_het': np.sum(np.mean(xss_concs[:, :, 8:8 + len(circuit_genes)], axis=0), axis=1),  # heterologous mRNA
            'p_het': np.sum(np.mean(xss_concs[:, :, 8 + len(circuit_genes):8 + len(circuit_genes) * 2], axis=0),
                            axis=1),
            # heterologous protein
        })

        # PLOT mRNA CONCENTRATIONS
        mRNA_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="mRNA conc., nM",
            x_range=tspan,
            title='mRNA concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            mRNA_figure.line(x='t', y='m_a', source=sources[i], line_width=1.5, line_color=self.gene_colours['a'],
                             legend_label='m_a', line_alpha=simtraj_alpha)  # plot metabolic mRNA concentrations
            mRNA_figure.line(x='t', y='m_r', source=sources[i], line_width=1.5, line_color=self.gene_colours['r'],
                             legend_label='m_r', line_alpha=simtraj_alpha)  # plot ribosomal mRNA concentrations
            mRNA_figure.line(x='t', y='m_het', source=sources[i], line_width=1.5, line_color=self.gene_colours['het'],
                             legend_label='m_het', line_alpha=simtraj_alpha)  # plot heterologous mRNA concentrations
        # plot average trajectory
        mRNA_figure.line(x='t', y='m_a', source=source_avg, line_width=1.5, line_color=self.gene_colours['a'],
                         legend_label='m_a')  # plot metabolic mRNA concentrations
        mRNA_figure.line(x='t', y='m_r', source=source_avg, line_width=1.5, line_color=self.gene_colours['r'],
                         legend_label='m_r')  # plot ribosomal mRNA concentrations
        mRNA_figure.line(x='t', y='m_het', source=source_avg, line_width=1.5, line_color=self.gene_colours['het'],
                         legend_label='m_het')  # plot heterologous mRNA concentrations
        # add and format the legend
        mRNA_figure.legend.label_text_font_size = "8pt"
        mRNA_figure.legend.location = "top_right"
        mRNA_figure.legend.click_policy = 'hide'

        # PLOT protein CONCENTRATIONS
        protein_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Protein conc., nM",
            x_range=tspan,
            title='Protein concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            protein_figure.line(x='t', y='p_a', source=sources[i], line_width=1.5, line_color=self.gene_colours['a'],
                                legend_label='p_a', line_alpha=simtraj_alpha)
            protein_figure.line(x='t', y='R', source=sources[i], line_width=1.5, line_color=self.gene_colours['r'],
                                legend_label='R', line_alpha=simtraj_alpha)
            protein_figure.line(x='t', y='p_het', source=sources[i], line_width=1.5,
                                line_color=self.gene_colours['het'],
                                legend_label='p_het', line_alpha=simtraj_alpha)
        # plot average trajectory
        protein_figure.line(x='t', y='p_a', source=source_avg, line_width=1.5, line_color=self.gene_colours['a'],
                            legend_label='p_a')
        protein_figure.line(x='t', y='R', source=source_avg, line_width=1.5, line_color=self.gene_colours['r'],
                            legend_label='R')
        protein_figure.line(x='t', y='p_het', source=source_avg, line_width=1.5, line_color=self.gene_colours['het'],
                            legend_label='p_het')
        # add and format the legend
        protein_figure.legend.label_text_font_size = "8pt"
        protein_figure.legend.location = "top_right"
        protein_figure.legend.click_policy = 'hide'

        # PLOT tRNA CONCENTRATIONS
        tRNA_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA conc., nM",
            x_range=tspan,
            title='tRNA concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            tRNA_figure.line(x='t', y='tc', source=sources[i], line_width=1.5, line_color=self.tRNA_colours['tc'],
                             legend_label='tc', line_alpha=simtraj_alpha)
            tRNA_figure.line(x='t', y='tu', source=sources[i], line_width=1.5, line_color=self.tRNA_colours['tu'],
                             legend_label='tu', line_alpha=simtraj_alpha)
        # plot average trajectory
        tRNA_figure.line(x='t', y='tc', source=source_avg, line_width=1.5, line_color=self.tRNA_colours['tc'],
                         legend_label='tc')
        tRNA_figure.line(x='t', y='tu', source=source_avg, line_width=1.5, line_color=self.tRNA_colours['tu'],
                         legend_label='tu')
        # add and format the legend
        tRNA_figure.legend.label_text_font_size = "8pt"
        tRNA_figure.legend.location = "top_right"
        tRNA_figure.legend.click_policy = 'hide'

        # PLOT INTRACELLULAR CHLORAMPHENICOL CONCENTRATION
        h_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="h, nM",
            x_range=tspan,
            title='Intracellular chloramphenicol concentration',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            h_figure.line(x='t', y='h', source=sources[i], line_width=1.5, line_color=self.gene_colours['h'],
                          legend_label='h', line_alpha=simtraj_alpha)
        # plot average trajectory
        h_figure.line(x='t', y='h', source=source_avg, line_width=1.5, line_color=self.gene_colours['h'],
                      legend_label='h')
        # add and format the legend
        h_figure.legend.label_text_font_size = "8pt"
        h_figure.legend.location = "top_right"
        h_figure.legend.click_policy = 'hide'

        return mRNA_figure, protein_figure, tRNA_figure, h_figure

    # plot concentrations for the synthetic circuits
    def plot_circuit_concentrations_multiple(self, ts, xss,
                                             par, circuit_genes, circuit_miscs, circuit_name2pos,
                                             # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                             circuit_styles,  # colours for the circuit plots
                                             dimensions=(320, 180), tspan=None,
                                             simtraj_alpha=0.1,
                                             varvol=False  # whether the simulation considers variable cell volumes
                                             ):
        # if considering variable cell volumes, divide heterologous molecule counts by cell volume
        if varvol:
            xss_concs = np.zeros_like(xss)  # initialise
            Vs = np.zeros_like(xss[:, :, 6])  # initialise
            for i in range(0, len(xss)):
                Vs[i, :] = xss[i, :, 6]  # cell volumes
                xss_concs[i, :, :] = np.divide(xss, (Vs[i, :] * np.ones_like(
                    np.array([xss[i, 0, :]]).T)).T)  # divide abundances by cell volumes to get concentrations
                xss_concs[i, :, 6] = par['s'] * np.ones_like(
                    Vs[i, :])  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xss_concs = xss

        # if no circuitry at all, return no plots
        if (len(circuit_genes) + len(circuit_miscs) == 0):
            return None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss_concs)):
            # Create a ColumnDataSource object for the plot
            data_for_column = {'t': ts}
            # record synthetic mRNA and protein concentrations
            for j in range(0, len(circuit_genes)):
                data_for_column['m_' + circuit_genes[j]] = xss_concs[i, :, 8 + j]
                data_for_column['p_' + circuit_genes[j]] = xss_concs[i, :, 8 + len(circuit_genes) + j]
            # record miscellaneous species' concentrations
            for j in range(0, len(circuit_miscs)):
                data_for_column[circuit_miscs[j]] = xss_concs[i, :, 8 + len(circuit_genes) * 2 + j]
            sources[i] = bkmodels.ColumnDataSource(data=data_for_column)

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts}
        # record synthetic mRNA and protein concentrations
        for j in range(0, len(circuit_genes)):
            data_for_column['m_' + circuit_genes[j]] = np.mean(xss_concs[:, :, 8 + j], axis=0)
            data_for_column['p_' + circuit_genes[j]] = np.mean(xss_concs[:, :, 8 + len(circuit_genes) + j], axis=0)
        # record miscellaneous species' concentrations
        for j in range(0, len(circuit_miscs)):
            data_for_column[circuit_miscs[j]] = np.mean(xss_concs[:, :, 8 + len(circuit_genes) * 2 + j], axis=0)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT mRNA and PROTEIN CONCENTRATIONS (IF ANY)
        if (len(circuit_genes) > 0):
            # mRNAs
            mRNA_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="mRNA conc., nM",
                x_range=tspan,
                title='mRNA concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            # plot simulated trajectories
            for i in range(0, len(xss_concs)):
                for gene in circuit_genes:
                    mRNA_figure.line(x='t', y='m_' + gene, source=sources[i], line_width=1.5,
                                     line_color=circuit_styles['colours'][gene],
                                     line_dash=circuit_styles['dashes'][gene],
                                     legend_label='m_' + gene, line_alpha=simtraj_alpha)
            # plot average trajectory
            for gene in circuit_genes:
                mRNA_figure.line(x='t', y='m_' + gene, source=source_avg, line_width=1.5,
                                 line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                                 legend_label='m_' + gene)
            # add and format the legend
            mRNA_figure.legend.label_text_font_size = "8pt"
            mRNA_figure.legend.location = 'top_left'
            mRNA_figure.legend.click_policy = 'hide'

            # proteins
            protein_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Protein conc., nM",
                x_range=tspan,
                title='Protein concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            # plot simulated trajectories
            for i in range(0, len(xss_concs)):
                for gene in circuit_genes:
                    protein_figure.line(x='t', y='p_' + gene, source=sources[i], line_width=1.5,
                                        line_color=circuit_styles['colours'][gene],
                                        line_dash=circuit_styles['dashes'][gene],
                                        legend_label='p_' + gene, line_alpha=simtraj_alpha)
            # plot average trajectory
            for gene in circuit_genes:
                protein_figure.line(x='t', y='p_' + gene, source=source_avg, line_width=1.5,
                                    line_color=circuit_styles['colours'][gene],
                                    line_dash=circuit_styles['dashes'][gene],
                                    legend_label='p_' + gene)
            # add and format the legend
            protein_figure.legend.label_text_font_size = "8pt"
            protein_figure.legend.location = 'top_left'
            protein_figure.legend.click_policy = 'hide'
        else:
            mRNA_figure = None
            protein_figure = None

        # PLOT MISCELLANEOUS SPECIES' CONCENTRATIONS (IF ANY)
        if (len(circuit_miscs) > 0):
            misc_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Conc., nM",
                x_range=tspan,
                title='Miscellaneous species concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            # plot simulated trajectories
            for i in range(0, len(xss_concs)):
                for misc in circuit_miscs:
                    misc_figure.line(x='t', y=misc, source=sources[i], line_width=1.5,
                                     line_color=circuit_styles['colours'][misc],
                                     line_dash=circuit_styles['dashes'][misc],
                                     legend_label=misc, line_alpha=simtraj_alpha)
            # plot average trajectory
            for misc in circuit_miscs:
                misc_figure.line(x='t', y=misc, source=source_avg, line_width=1.5,
                                 line_color=circuit_styles['colours'][misc], line_dash=circuit_styles['dashes'][misc],
                                 legend_label=misc)
            # add and format the legend
            misc_figure.legend.label_text_font_size = "8pt"
            misc_figure.legend.location = 'top_left'
            misc_figure.legend.click_policy = 'hide'
        else:
            misc_figure = None

        return mRNA_figure, protein_figure, misc_figure

    # plot transcription regulation function values for the circuit's genes
    def plot_circuit_regulation_multiple(self, ts, xss,
                                         par, circuit_F_calc,
                                         # function calculating the transcription regulation functions for the circuit
                                         circuit_genes, circuit_miscs, circuit_name2pos,
                                         # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                         circuit_styles,  # colours for the circuit plots
                                         dimensions=(320, 180), tspan=None,
                                         simtraj_alpha=0.1,
                                         varvol=False  # whether the simulation considers variable cell volumes
                                         ):
        # if considering variable cell volumes, divide heterologous molecule counts by cell volume
        if varvol:
            xss_concs = np.zeros_like(xss)  # initialise
            Vs = np.zeros_like(xss[:, :, 6])  # initialise
            for i in range(0, len(xss)):
                Vs[i, :] = xss[i, :, 6]  # cell volumes
                xss_concs[i, :, :] = np.divide(xss, (Vs[i, :] * np.ones_like(
                    np.array([xss[i, 0, :]]).T)).T)  # divide abundances by cell volumes to get concentrations
                xss_concs[i, :, 6] = par['s'] * np.ones_like(
                    Vs[i, :])  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xss_concs = xss

        # if no circuitry at all, return no plots
        if (len(circuit_genes) + len(circuit_miscs) == 0):
            return None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # find values of gene transcription regulation functions and create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss_concs)):
            Fs = np.zeros((len(ts), len(circuit_genes)))  # initialise
            for k in range(0, len(ts)):
                Fs[k, :] = np.array(circuit_F_calc(ts[k], xss_concs[i, k, :], par, circuit_name2pos)[:])

            # Create a ColumnDataSource object for the plot
            data_for_column = {'t': ts}
            for j in range(0, len(circuit_genes)):
                data_for_column['F_' + circuit_genes[j]] = Fs[:, j]
            sources[i] = bkmodels.ColumnDataSource(data=data_for_column)

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts}
        for j in range(0, len(circuit_genes)):
            data_for_column['F_' + circuit_genes[j]] = np.zeros_like(ts)
        # add gene transcription regulation functions for different trajectories together
        for i in range(0, len(xss_concs)):
            for j in range(0, len(circuit_genes)):
                data_for_column['F_' + circuit_genes[j]] += np.array(sources[i].data['F_' + circuit_genes[j]])
        # divide by the number of trajectories to get the average
        for j in range(0, len(circuit_genes)):
            data_for_column['F_' + circuit_genes[j]] /= len(xss_concs)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT TRANSCRIPTION REGULATION FUNCTIONS
        F_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Transc. reg. funcs. F",
            x_range=tspan,
            y_range=(0, 1.05),
            title='Gene transcription regulation',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            for gene in circuit_genes:
                F_figure.line(x='t', y='F_' + gene, source=sources[i], line_width=1.5,
                              line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                              legend_label='F_' + gene, line_alpha=simtraj_alpha)
        # plot average trajectory
        for gene in circuit_genes:
            F_figure.line(x='t', y='F_' + gene, source=source_avg, line_width=1.5,
                          line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                          legend_label='F_' + gene)
        # add and format the legend
        F_figure.legend.label_text_font_size = "8pt"
        F_figure.legend.location = 'top_left'
        F_figure.legend.click_policy = 'hide'

        return F_figure

    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables_multiple(self, ts, xss,
                                     par, circuit_genes, circuit_miscs, circuit_name2pos,
                                     # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                     circuit_eff_m_het_div_k_het,
                                     # function calculating the total effective mRNA conc./k value for all heterologous genes
                                     dimensions=(320, 180), tspan=None,
                                     simtraj_alpha=0.1,
                                     varvol=False  # whether the simulation considers variable cell volumes
                                     ):
        # if considering variable cell volumes, divide heterologous molecule counts by cell volume
        if varvol:
            xss_concs = np.zeros_like(xss)  # initialise
            Vs = np.zeros_like(xss[:, :, 6])  # initialise
            for i in range(0, len(xss)):
                Vs[i, :] = xss[i, :, 6]  # cell volumes
                xss_concs[i, :, :] = np.divide(xss, (Vs[i, :] * np.ones_like(
                    np.array([xss[i, 0, :]]).T)).T)  # divide abundances by cell volumes to get concentrations
                xss_concs[i, :, 6] = par['s'] * np.ones_like(
                    Vs[i, :])  # instead of volumes, x without variable volumes has nutrient quality in this position
        else:
            xss_concs = xss

        # if no circuitry at all, return no plots
        if (len(circuit_genes) + len(circuit_miscs) == 0):
            return None, None, None, None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time and create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss_concs)):
            e, l, F_r, nu, psi, T, D, D_nodeg = self.get_e_l_Fr_nu_psi_T_D_Dnodeg(ts, xss_concs[i, :, :],
                                                                                  par, circuit_genes, circuit_miscs,
                                                                                  circuit_name2pos,
                                                                                  circuit_eff_m_het_div_k_het)
            # Create a ColumnDataSource object for the plot
            sources[i] = bkmodels.ColumnDataSource(data={
                't': np.array(ts),
                'e': np.array(e),
                'l': np.array(l),
                'F_r': np.array(F_r),
                'nu': np.array(nu),
                'psi': np.array(psi),
                '1/T': np.array(1 / T),
                'D': np.array(D),
                'D_nodeg': np.array(D_nodeg)
            })

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts,
                           'e': np.zeros_like(ts),
                           'l': np.zeros_like(ts),
                           'F_r': np.zeros_like(ts),
                           'nu': np.zeros_like(ts),
                           'psi': np.zeros_like(ts),
                           '1/T': np.zeros_like(ts),
                           'D': np.zeros_like(ts),
                           'D_nodeg': np.zeros_like(ts)}
        # add physiological variables for different trajectories together
        for i in range(0, len(xss_concs)):
            data_for_column['e'] += np.array(sources[i].data['e'])
            data_for_column['l'] += np.array(sources[i].data['l'])
            data_for_column['F_r'] += np.array(sources[i].data['F_r'])
            data_for_column['nu'] += np.array(sources[i].data['nu'])
            data_for_column['psi'] += np.array(sources[i].data['psi'])
            data_for_column['1/T'] += np.array(sources[i].data['1/T'])
            data_for_column['D'] += np.array(sources[i].data['D'])
            data_for_column['D_nodeg'] += np.array(sources[i].data['D_nodeg'])
        # divide by the number of trajectories to get the average
        data_for_column['e'] /= len(xss_concs)
        data_for_column['l'] /= len(xss_concs)
        data_for_column['F_r'] /= len(xss_concs)
        data_for_column['nu'] /= len(xss_concs)
        data_for_column['psi'] /= len(xss_concs)
        data_for_column['1/T'] /= len(xss_concs)
        data_for_column['D'] /= len(xss_concs)
        data_for_column['D_nodeg'] /= len(xss_concs)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT GROWTH RATE
        l_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Growth rate, 1/h",
            x_range=tspan,
            title='Growth rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            l_figure.line(x='t', y='l', source=sources[i], line_width=1.5, line_color='blue', legend_label='l',
                          line_alpha=simtraj_alpha)
        # plot average trajectory
        l_figure.line(x='t', y='l', source=source_avg, line_width=1.5, line_color='blue', legend_label='l')
        # add and format the legend
        l_figure.legend.label_text_font_size = "8pt"
        l_figure.legend.location = 'top_left'
        l_figure.legend.click_policy = 'hide'

        # PLOT TRANSLATION ELONGATION RATE
        e_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Translation elongation rate, aa/s",
            x_range=tspan,
            title='Translation elongation rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            e_figure.line(x='t', y='e', source=sources[i], line_width=1.5, line_color='blue', legend_label='e',
                          line_alpha=simtraj_alpha)
        # plot average trajectory
        e_figure.line(x='t', y='e', source=source_avg, line_width=1.5, line_color='blue', legend_label='e')
        # add and format the legend
        e_figure.legend.label_text_font_size = "8pt"
        e_figure.legend.location = 'top_left'
        e_figure.legend.click_policy = 'hide'

        # PLOT RIBOSOMAL GENE TRANSCRIPTION REGULATION FUNCTION
        F_r_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Ribosomal gene transc. reg. func. F_r",
            x_range=tspan,
            y_range=(0, 1.05),
            title='Ribosomal gene transcription regulation function',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            F_r_figure.line(x='t', y='F_r', source=sources[i], line_width=1.5, line_color='blue', legend_label='F_r',
                            line_alpha=simtraj_alpha)
        # plot average trajectory
        F_r_figure.line(x='t', y='F_r', source=source_avg, line_width=1.5, line_color='blue', legend_label='F_r')
        # add and format the legend
        F_r_figure.legend.label_text_font_size = "8pt"
        F_r_figure.legend.location = 'top_left'
        F_r_figure.legend.click_policy = 'hide'

        # PLOT ppGpp CONCENTRATION
        ppGpp_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="[ppGpp], nM",
            x_range=tspan,
            title='ppGpp concentration',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            ppGpp_figure.line(x='t', y='1/T', source=sources[i], line_width=1.5, line_color='blue', legend_label='1/T',
                              line_alpha=simtraj_alpha)
        # plot average trajectory
        ppGpp_figure.line(x='t', y='1/T', source=source_avg, line_width=1.5, line_color='blue', legend_label='1/T')
        # add and format the legend
        ppGpp_figure.legend.label_text_font_size = "8pt"
        ppGpp_figure.legend.location = 'top_left'
        ppGpp_figure.legend.click_policy = 'hide'

        # PLOT tRNA CHARGING RATE
        nu_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA charging rate, 1/s",
            x_range=tspan,
            title='tRNA charging rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            nu_figure.line(x='t', y='nu', source=sources[i], line_width=1.5, line_color='blue', legend_label='nu',
                           line_alpha=simtraj_alpha)
        # plot average trajectory
        nu_figure.line(x='t', y='nu', source=source_avg, line_width=1.5, line_color='blue', legend_label='nu')

        # PLOT RC DENOMINATOR
        D_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="RC denominator",
            x_range=tspan,
            title='RC denominator',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss_concs)):
            D_figure.line(x='t', y='D', source=sources[i], line_width=1.5, line_color='blue', legend_label='D',
                          line_alpha=simtraj_alpha)
            # D_figure.line(x='t', y='D_nodeg', source=sources[i], line_width=1.5, line_color='red', legend_label='D_nodeg', line_alpha=simtraj_alpha)
        # plot average trajectory
        D_figure.line(x='t', y='D', source=source_avg, line_width=1.5, line_color='blue', legend_label='D')
        # D_figure.line(x='t', y='D_nodeg', source=source_avg, line_width=1.5, line_color='red', legend_label='D_nodeg')
        # add and format the legend
        D_figure.legend.label_text_font_size = "8pt"
        D_figure.legend.location = 'top_left'
        D_figure.legend.click_policy = 'hide'

        return l_figure, e_figure, F_r_figure, ppGpp_figure, nu_figure, D_figure


# DETERMINISTIC SIMULATION ---------------------------------------------------------------------------------------------
# ODE simulator with DIFFRAX
@functools.partial(jax.jit, static_argnums=(1, 3, 4))
def ode_sim(par,  # dictionary with model parameters
            ode_with_circuit,  # ODE function for the cell with the synthetic gene circuit
            x0,  # initial condition VECTOR
            num_circuit_genes, num_circuit_miscs, circuit_name2pos, sgp4j,
            # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder, relevant synthetic gene parameters in jax.array form
            tf, ts, rtol, atol,
            # simulation parameters: time frame, when to save the system's state, relative and absolute tolerances
            solver=Kvaerno3()  # ODE solver
            ):
    # define the ODE term
    vector_field = lambda t, y, args: ode_with_circuit(t, y, args)
    term = ODETerm(vector_field)

    # define arguments of the ODE term
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        num_circuit_genes, num_circuit_miscs,  # number of genes and miscellaneous species in the circuit
        sgp4j  # relevant synthetic gene parameters in jax.array form
    )

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
def ode(t, x,
        circuit_ode, circuit_eff_m_het_div_k_het,
        args):
    # unpack the args
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]  # number of genes in the circuit
    num_circuit_miscs = args[3]  # number of miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het, g_het = args[4]  # unpack jax-arrayed synthetic gene parameters

    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    s = x[6]  # nutrient quality (constant)
    h = x[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with circuit_name2pos

    # FIND SPECIAL SYNTHETIC PROTEIN CONCENTRATIONS - IF PRESENT
    # chloramphenicol acetyltransferase (antibiotic reistance)
    p_cat = jax.lax.select(par['cat_gene_present'] == 1, x[circuit_name2pos['p_cat']], 0.0)
    # synthetic protease (synthetic protein degradation)
    p_prot = jax.lax.select(par['prot_gene_present'] == 1, x[circuit_name2pos['p_prot']], 0.0)

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])
    k_het = k_calc(e, kplus_het, kminus_het, n_het)

    T = tc / tu  # ratio of charged to uncharged tRNAs

    H = (par['K_D'] + h) / par['K_D']  # corection to ribosome availability due to chloramphenicol action

    # heterologous mRNA levels scaled by RBS strength
    m_het_div_k_het = circuit_eff_m_het_div_k_het(x, par, circuit_name2pos, num_circuit_genes, e, k_het)

    # heterologous protein degradation flux
    prodeflux = jnp.sum(
        # (degradation rate times protease level times protein concnetration) times number of AAs per protein
        d_het * p_prot * x[8 + num_circuit_genes:8 + num_circuit_genes * 2] * n_het
    )
    prodeflux_times_H_div_eR = prodeflux * H / (e * R)  # degradation flux scaled by overall protein synthesis rate

    # resource competition denominator
    m_notq_div_k_notq = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_times_H_div_eR) * m_notq_div_k_notq - par[
        'phi_q'] * prodeflux_times_H_div_eR) / \
                (1 - par['phi_q'] * (1 - prodeflux_times_H_div_eR))
    D = H * (1 + mq_div_kq + m_notq_div_k_notq)
    B = R * (1 / H - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    metab_burd = jnp.sum(g_het * x[
                                 8 + num_circuit_genes:8 + num_circuit_genes * 2])  # total metabolic burden imposed by synthetic proteins

    nu = nu_calc(par, tu, s, metab_burd)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # return dx/dt for the host cell
    dxdt = jnp.array([
                         # mRNAs
                         l * par['c_a'] * par['a_a'] - (par['b_a'] + l) * m_a,
                         l * Fr_calc(par, T) * par['c_r'] * par['a_r'] - (par['b_r'] + l) * m_r,
                         # metabolic protein p_a
                         (e / par['n_a']) * (m_a / k_a / D) * R - l * p_a,
                         # ribosomes
                         (e / par['n_r']) * (m_r / k_r / D) * R - l * R,
                         # tRNAs
                         nu * p_a - l * tc - e * B,
                         l * psi - l * tu - nu * p_a + e * B,
                         # nutrient quality assumed constant
                         0,
                         # chloramphenicol concentration
                         par['diff_h'] * (par['h_ext'] - h) - h * p_cat / par['K_C'] - l * h - par['eff_h'] * h,
                     ] +
                     circuit_ode(t, x, e, l, R, k_het, D, p_prot,
                                 par, circuit_name2pos)
                     )
    return dxdt


# TAU-LEAPING SIMULATION -----------------------------------------------------------------------------------------------
def tauleap_sim(par,  # dictionary with model parameters
                circuit_v,  # calculating the propensity vector for stochastic simulation of circuit expression
                circuit_eff_m_het_div_k_het,
                # calculating the effective mRNA/k values for the synthetic genes in the circuit
                x0,  # initial condition VECTOR
                num_circuit_genes, num_circuit_miscs, circuit_name2pos,
                # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder,
                sgp4j,  # relevant synthetic gene parameters in jax.array form
                tf, tau, tau_odestep, tau_savetimestep,
                # simulation parameters: time frame, tau-leap time step, number of ODE steps in each tau-leap step
                mRNA_count_scales, S, circuit_synpos2genename, keys0,
                # parameter vectors for efficient simulation and reaction stoichiometry info - determined by tauleap_sim_prep!!
                avg_dynamics=False
                # true if considering the deterministic cas with average dynamics of random variables
                ):
    # define the arguments for finding the next state vector
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        num_circuit_genes, num_circuit_miscs,  # number of genes and miscellaneous species in the circuit
        sgp4j,  # relevant synthetic gene parameters in jax.array form
        mRNA_count_scales, S, circuit_synpos2genename  # parameters for stochastic simulation
    )

    # time points at which we save the solution
    ts = jnp.arange(tf[0], tf[1] + tau_savetimestep / 2, tau_savetimestep)

    # number of ODE steps in each tau-leap step
    ode_steps_in_tau = int(tau / tau_odestep)

    # make the retrieval of next x a lambda-function for jax.lax.scanning
    scan_step = lambda sim_state, t: tauleap_record_x(circuit_v, circuit_eff_m_het_div_k_het,
                                                      sim_state, t, tau, ode_steps_in_tau,
                                                      args)

    # define the jac.lax.scan function
    tauleap_scan = lambda sim_state_rec0, ts: jax.lax.scan(scan_step, sim_state_rec0, ts)
    tauleap_scan_jit = jax.jit(tauleap_scan)

    # get initial conditions
    if (len(x0.shape) == 1):  # if x0 common for all trajectories, copy it for each trajectory
        x0s = jnp.repeat(x0.reshape((1, x0.shape[0])), keys0.shape[0], axis=0)
    else:  # otherwise, leave initial conditions as is
        x0s = x0

    # initalise the simulator state: (t, x, sim_step_cntr, record_step_cntr, key, tf, xs)
    sim_state_rec0 = {'t': tf[0], 'x': x0s,  # time, state vector
                      'key': keys0,  # random number generator key
                      'tf': tf,  # overall simulation time frame
                      'save_every_n_steps': int(tau_savetimestep / tau),  # tau-leap steps between record points
                      'avg_dynamics': avg_dynamics
                      # true if considering the deterministic cas with average dynamics of random variables
                      }

    # vmapping - specify that we vmap over the random number generation keys
    sim_state_vmap_axes = {'t': None, 'x': 0,  # time, state vector
                           'key': 0,  # random number generator key
                           'tf': None,  # overall simulation time frame
                           'save_every_n_steps': None,  # tau-leap steps between record points
                           'avg_dynamics': None
                           # true if considering the deterministic cas with average dynamics of random variables
                           }

    # simulate (with vmapping)
    sim_state_rec_final, xs = jax.jit(jax.vmap(tauleap_scan_jit, (sim_state_vmap_axes, None)))(sim_state_rec0, ts)

    return ts, xs, sim_state_rec_final['key']


# log the next trajectory point
def tauleap_record_x(circuit_v,  # calculating the propensity vector for stochastic simulation of circuit expression
                     circuit_eff_m_het_div_k_het,
                     # calculating the effective mRNA/k values for the synthetic genes in the circuit
                     sim_state_record,  # simulator state
                     t,  # time of last record
                     tau,  # time step
                     ode_steps_in_tau,  # number of ODE integration steps in each tau-leap step
                     args):
    # DEFINITION OF THE ACTUAL TAU-LEAP SIMULATION STEP
    def tauleap_next_x(step_cntr, sim_state_tauleap):
        # update t
        next_t = sim_state_tauleap['t'] + tau

        # update x
        # find deterministic change in x
        det_update = tauleap_integrate_ode(sim_state_tauleap['t'], sim_state_tauleap['x'], tau, ode_steps_in_tau,
                                           circuit_eff_m_het_div_k_het,
                                           args)
        # find stochastic change in x
        stoch_update = tauleap_update_stochastically(sim_state_tauleap['t'], sim_state_tauleap['x'],
                                                     tau, args, circuit_v,
                                                     sim_state_tauleap['key'], sim_state_tauleap['avg_dynamics'])
        # find next x
        next_x_tentative = sim_state_tauleap['x'] + det_update + stoch_update
        # make sure x has no negative entries
        next_x = jax.lax.select(next_x_tentative < 0, jnp.zeros_like(next_x_tentative), next_x_tentative)
        # next_x=jnp.multiply(next_x_tentative>=0,next_x_tentative)

        # update key
        next_key, _ = jax.random.split(sim_state_tauleap['key'], 2)

        return {
            # entries updated over the course of the tau-leap step
            't': next_t, 'x': next_x,
            'key': next_key,
            # entries unchanged over the course of the tau-leap step
            'tf': sim_state_tauleap['tf'],
            'save_every_n_steps': sim_state_tauleap['save_every_n_steps'],
            'avg_dynamics': sim_state_tauleap['avg_dynamics']
        }

    # FUNCTION BODY
    # run tau-leap integration until the next state is to be saved
    next_state_bytauleap = jax.lax.fori_loop(0, sim_state_record['save_every_n_steps'], tauleap_next_x,
                                             sim_state_record)

    # update the overall simulator state
    next_sim_state_record = {
        # entries updated over the course of the tau-leap step
        't': next_state_bytauleap['t'], 'x': next_state_bytauleap['x'],
        'key': next_state_bytauleap['key'],
        # entries unchanged
        'tf': sim_state_record['tf'],
        'save_every_n_steps': sim_state_record['save_every_n_steps'],
        'avg_dynamics': sim_state_record['avg_dynamics']
    }

    return next_sim_state_record, sim_state_record['x']


# ode integration - Euler method
def tauleap_integrate_ode(t, x, tau, ode_steps_in_tau,
                          circuit_eff_m_het_div_k_het,
                          args):
    # def euler_step(ode_step, x):
    #     return x + tauleap_ode(t + ode_step_size * ode_step, x, args) * ode_step_size
    #
    # ode_step_size = tau / ode_steps_in_tau
    # # integrate the ODE
    # x_new= jax.lax.fori_loop(0, ode_steps_in_tau, euler_step, x)
    #
    # return x_new - x
    return tauleap_ode(t, x, circuit_eff_m_het_div_k_het, args) * tau


# ode for the deterministic part of the tau-leaping simulation
def tauleap_ode(t, x, circuit_eff_m_het_div_k_het, args):
    # unpack the args
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]  # number of genes in the circuit
    num_circuit_miscs = args[3]  # number of miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het, g_het = args[4]  # unpack jax-arrayed synthetic gene parameters

    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    s = x[6]  # nutrient quality (constant)
    h = x[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with circuit_name2pos

    # FIND SPECIAL SYNTHETIC PROTEIN CONCENTRATIONS - IF PRESENT
    # chloramphenicol acetyltransferase (antibiotic reistance)
    p_cat = jax.lax.select(par['cat_gene_present'] == 1, x[circuit_name2pos['p_cat']], 0.0)
    # synthetic protease (synthetic protein degradation)
    p_prot = jax.lax.select(par['prot_gene_present'] == 1, x[circuit_name2pos['p_prot']], 0.0)

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])
    k_het = k_calc(e, kplus_het, kminus_het, n_het)

    # ratio of charged to uncharged tRNAs
    T = tc / tu

    # corection to ribosome availability due to chloramphenicol action
    H = (par['K_D'] + h) / par['K_D']

    # heterologous mRNA levels scaled by RBS strength
    m_het_div_k_het = circuit_eff_m_het_div_k_het(x, par, circuit_name2pos, num_circuit_genes, e, k_het)

    # heterologous protein degradation flux
    prodeflux = jnp.sum(
        # (degradation rate times protease level times protein concnetration) times number of AAs per protein
        d_het * p_prot * x[8 + num_circuit_genes:8 + num_circuit_genes * 2] * n_het
    )
    prodeflux_times_H_div_eR = prodeflux * H / (e * R)  # degradation flux scaled by overall protein synthesis rate

    # resource competition denominator
    m_notq_div_k_notq = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_times_H_div_eR) * m_notq_div_k_notq - par[
        'phi_q'] * prodeflux_times_H_div_eR) / \
                (1 - par['phi_q'] * (1 - prodeflux_times_H_div_eR))
    D = H * (1 + mq_div_kq + m_notq_div_k_notq)
    B = R * (1 / H - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    metab_burd = jnp.sum(
        g_het * x[8 + num_circuit_genes:8 + num_circuit_genes * 2])  # metabolic burden imposed by synthetic proteins

    nu = nu_calc(par, tu, s, metab_burd)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # continuously, we only consider charged aa-tRNA consumption by NATIVE protein translation
    B_cont = R * (1 / H - 1 / D - jnp.sum(jnp.divide(x[8:8 + num_circuit_genes], k_het)) / D)

    # return dx/dt for the host cell
    return jnp.array([
                         # mRNAs
                         l * par['c_a'] * par['a_a'] - (par['b_a'] + l) * m_a,
                         l * Fr_calc(par, T) * par['c_r'] * par['a_r'] - (par['b_r'] + l) * m_r,
                         # metabolic protein p_a
                         (e / par['n_a']) * (m_a / k_a / D) * R - l * p_a,
                         # ribosomes
                         (e / par['n_r']) * (m_r / k_r / D) * R - l * R,
                         # tRNAs
                         nu * p_a - l * tc - e * B_cont,
                         l * psi - l * tu - nu * p_a + e * B_cont,
                         # nutrient quality assumed constant
                         0,
                         # chloramphenicol concentration
                         par['diff_h'] * (par['h_ext'] - h) - h * p_cat / par['K_C'] - l * h
                     ] +
                     [0] * (2 * num_circuit_genes) + [0] * num_circuit_miscs
                     # synthetic gene expression considered stochastically
                     )


def tauleap_update_stochastically(t, x, tau, args, circuit_v,
                                  key, avg_dynamics):
    # PREPARATION
    # unpack the arguments
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]  # number of genes in the circuit
    num_circuit_miscs = args[3]  # number of miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het, g_het = args[4]  # unpack jax-arrayed synthetic gene parameters
    # stochastic simulation arguments
    mRNA_count_scales = args[5]
    S = args[6]
    circuit_synpos2genename = args[7]  # parameter vectors for efficient simulation and reaction stoichiometry info

    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    s = x[6]  # nutrient quality (constant)
    h = x[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with circuit_name2pos

    # FIND SPECIAL SYNTHETIC PROTEIN CONCENTRATIONS - IF PRESENT
    # chloramphenicol acetyltransferase (antibiotic reistance)
    p_cat = jax.lax.select(par['cat_gene_present'] == 1, x[circuit_name2pos['p_cat']], 0.0)
    # synthetic protease (synthetic protein degradation)
    p_prot = jax.lax.select(par['prot_gene_present'] == 1, x[circuit_name2pos['p_prot']], 0.0)

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])
    k_het = k_calc(e, kplus_het, kminus_het, n_het)

    T = tc / tu  # ratio of charged to uncharged tRNAs

    H = (par['K_D'] + h) / par['K_D']  # corection to ribosome availability due to chloramphenicol action

    m_het_div_k_het = jnp.sum(x[8:8 + num_circuit_genes] / k_het)  # heterologous protein synthesis flux
    prodeflux = jnp.sum(
        d_het * n_het * x[8 + num_circuit_genes:8 + num_circuit_genes * 2])  # heterologous protein degradation flux
    prodeflux_times_H_div_eR = prodeflux * H / (e * R)

    # resource competition denominator
    m_notq_div_k_notq = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_times_H_div_eR) * m_notq_div_k_notq - par[
        'phi_q'] * prodeflux_times_H_div_eR) / \
                (1 - par['phi_q'] * (1 - prodeflux_times_H_div_eR))
    D = H * (1 + mq_div_kq + m_notq_div_k_notq)
    B = R * (1 / H - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    metab_burd = jnp.sum(g_het * x[
                                 8 + num_circuit_genes:8 + num_circuit_genes * 2])  # total metabolic burden imposed by synthetic proteins

    nu = nu_calc(par, tu, s, metab_burd)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # continuously, we only consider charged aa-tRNA consumption by NATIVE protein translation
    B_cont = (D - 1 - jnp.sum(jnp.divide(x[8:8 + num_circuit_genes], k_het))) / D * R

    # FIND REACTION PROPENSITIES
    v = jnp.array(circuit_v(t, x,  # time, cell state, external inputs
                            e, l,  # translation elongation rate, growth rate
                            R,  # ribosome count in the cell
                            k_het, D,
                            # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
                            p_prot,  # synthetic protease concentration
                            mRNA_count_scales,  # scaling factors for synthetic gene mRNA counts
                            par,  # system parameters
                            circuit_name2pos
                            ))

    # RETURN THE NUMBER OF TIMES THAT EACH REACTION HAS OCCURRED
    # or take the average if we're considering the deterministic case
    return jax.lax.select(avg_dynamics, jnp.matmul(S, v * tau), jnp.matmul(S, jax.random.poisson(key=key, lam=v * tau)))


# preparatory step creating the objects necessary for tau-leap simulation - DO NOT JIT
def tauleap_sim_prep(par,  # dictionary with model parameters
                     num_circuit_genes, num_circuit_miscs, circuit_name2pos,
                     x0_det,  # initial condition found deterministically
                     key_seeds=[0]  # random key seed(s)
                     ):
    # DICTIONARY mapping positions of a synthetic gene in the list of all synthetic genes to their names
    circuit_synpos2genename = {}
    for name in circuit_name2pos.keys():
        if (name[0] == 'm'):
            circuit_synpos2genename[circuit_name2pos[name] - 8] = name[2:]

    # PARAMETER VECTORS FOR EFFICIENT SIMULATION
    # mRNA count scalings
    mRNA_count_scales_np = np.zeros((num_circuit_genes))
    for i in range(0, num_circuit_genes):
        mRNA_count_scales_np[i] = par['n_' + circuit_synpos2genename[i]] / 25
    mRNA_count_scales = jnp.array(mRNA_count_scales_np)

    # STOICHIOMETRY MATRIX
    S = gen_stoich_mat(par,
                       circuit_name2pos, circuit_synpos2genename,
                       num_circuit_genes, num_circuit_miscs,
                       mRNA_count_scales)

    # MAKE THE INITIAL CONDITION FOR TAU-LEAPING WITH APPROPRIATE INTEGER VALUES OF RANDOM VARIABLES
    x0_det_np = np.array(x0_det)  # convert to numpy array for convenience
    x0_tauleap_mrnas = np.multiply(np.round(x0_det_np[8:8 + num_circuit_genes] / mRNA_count_scales_np),
                                   mRNA_count_scales_np)  # synthetic mRNA counts - must be a multiple of the corresponding scaling factors that acount for translation by multiple ribosomes
    x0_tauleap_prots_and_miscs = np.round(
        x0_det_np[8 + num_circuit_genes:])  # synthetic protein and miscellaneous species counts
    x0_tauleap = np.concatenate((x0_det_np[0:8], x0_tauleap_mrnas, x0_tauleap_prots_and_miscs), axis=0)

    # STARTING RANDOM NUMBER GENERATION KEY
    key_seeds_jnp = jnp.array(key_seeds)
    keys0 = jax.vmap(jax.random.PRNGKey)(key_seeds_jnp)

    return mRNA_count_scales, S, x0_tauleap, circuit_synpos2genename, keys0


# generate the stocihiometry matrix for heterologous genes (plus return total number of stochastic reactions) - DO NOT JIT
def gen_stoich_mat(par,
                   circuit_name2pos, circuit_synpos2genename,
                   num_circuit_genes, num_circuit_miscs,
                   mRNA_count_scales
                   ):
    # unpack stochionmetry args
    mRNA_count_scales_np = np.array(mRNA_count_scales)

    # find the number of stochastic reactions that can occur
    num_stoch_reactions = num_circuit_genes * (3 +  # synthesis/degradation/dilution of mRNA
                                               3)  # synthesis/degradation/dilution of protein
    if ('cat_pb' in circuit_name2pos.keys()):  # plus, might need to model stochastic action of integrase
        num_stoch_reactions += 2 + 2  # functional CAT gene forward and reverse strain exchange, LR site-integrase dissociation due to conformation change and plasmid replication

    # initialise (in numpy format)
    S = np.zeros((8 + 2 * num_circuit_genes + num_circuit_miscs, num_stoch_reactions))

    # initialise thge counter of reactions in S
    reaction_cntr = 0

    # mRNA - reactions common for all genes
    for i in range(0, num_circuit_genes):
        S[8 + i, reaction_cntr] = mRNA_count_scales_np[i]  # mRNA synthesis
        reaction_cntr += 1
        S[8 + i, reaction_cntr] = -mRNA_count_scales_np[i]  # mRNA degradation
        reaction_cntr += 1
        S[8 + i, reaction_cntr] = -mRNA_count_scales_np[i]  # mRNA dilution
        reaction_cntr += 1

    # protein - reactions common for all genes
    for i in range(0, num_circuit_genes):
        S[8 + num_circuit_genes + i, reaction_cntr] = 1  # protein synthesis
        S[4, reaction_cntr] = -par[
            'n_' + circuit_synpos2genename[i]]  # includes tRNA unchraging during translation(-tc)
        S[5, reaction_cntr] = par['n_' + circuit_synpos2genename[i]]  # includes tRNA unchraging during translation(+tu)
        reaction_cntr += 1
        S[8 + num_circuit_genes + i, reaction_cntr] = -1  # protein degradation
        reaction_cntr += 1
        S[8 + num_circuit_genes + i, reaction_cntr] = -1  # protein dilution
        reaction_cntr += 1

    if ('cat_pb' in circuit_name2pos.keys()):  # stochastic action of integrase
        # CAT gene forward strain exchange: from cat_pb to cat_lri1
        S[8 + 2 * num_circuit_genes, reaction_cntr] = -1  # cat_pb decreased
        S[8 + 2 * num_circuit_genes + 1, reaction_cntr] = 1  # cat_lri1 increased
        reaction_cntr += 1

        # CAT gene reverse strain exchange: from cat_lri1 to cat_pb
        S[8 + 2 * num_circuit_genes, reaction_cntr] = 1  # cat_pb increased
        S[8 + 2 * num_circuit_genes + 1, reaction_cntr] = -1  # cat_lri1 decreased
        reaction_cntr += 1

        # LR site-integrase dissociation due to conformation change
        S[8 + 2 * num_circuit_genes + 1, reaction_cntr] = -1  # cat_lri1 decreased
        reaction_cntr += 1

        # LR site-integrase dissociation due to plasmid replication
        S[8 + 2 * num_circuit_genes + 1, reaction_cntr] = -1  # cat_lri1 decreased
        reaction_cntr += 1

    return jnp.array(S)


# TAU-LEAPING SIMULATION FOR SWITCHING DETECTION (MEMORY-EFFICIENT) ----------------------------------------------------
# simulate to find switching time
def tauleap_sim_switch(par,  # dictionary with model parameters
                       circuit_v,  # calculating the propensity vector for stochastic simulation of circuit expression
                       circuit_eff_m_het_div_k_het,
                       # calculating the effective mRNA/k values for the synthetic genes in the circuit
                       x0,  # initial condition VECTOR
                       num_circuit_genes, num_circuit_miscs, circuit_name2pos,
                       # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder,
                       sgp4j,  # relevant synthetic gene parameters in jax.array form
                       tf, tau, tau_odestep, tau_checkswitchstep,
                       # simulation parameters: time frame, tau-leap time step, number of ODE steps in each tau-leap step
                       mRNA_count_scales, S, circuit_synpos2genename, keys0,
                       # parameter vectors for efficient simulation and reaction stoichiometry info - determined by tauleap_sim_prep!!
                       switch_condition,  # function that returns true if the switching condition is met
                       switch_condition1  # additional function that returns true if the switching condition is met
                       ):
    # define the arguments for finding the next state vector
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        num_circuit_genes, num_circuit_miscs,  # number of genes and miscellaneous species in the circuit
        sgp4j,  # relevant synthetic gene parameters in jax.array form
        mRNA_count_scales, S, circuit_synpos2genename  # parameters for stochastic simulation
    )

    # time points at which we save the solution
    num_checkswitchsteps = int((tf[1] - tf[0]) / tau_checkswitchstep)

    # number of ODE steps in each tau-leap step
    ode_steps_in_tau = int(tau / tau_odestep)

    # make the retrieval of next x a lambda-function for jax.lax.scanning
    loop_step = lambda i, sim_state: tauleap_check_switch(circuit_v, circuit_eff_m_het_div_k_het,
                                                          sim_state, i, tau, ode_steps_in_tau, args,
                                                          switch_condition, switch_condition1)

    # define the jac.lax.scan function
    tauleap_loop = lambda sim_state_rec0: jax.lax.fori_loop(0, num_checkswitchsteps, loop_step, sim_state_rec0)
    tauleap_loop_jit = jax.jit(tauleap_loop)

    # get initial conditions
    if (len(x0.shape) == 1):  # if x0 common for all trajectories, copy it for each trajectory
        x0s = jnp.repeat(x0.reshape((1, x0.shape[0])), keys0.shape[0], axis=0)
    else:  # otherwise, leave initial conditions as is
        x0s = x0

    # initalise the simulator state: (t, x, sim_step_cntr, record_step_cntr, key, tf, xs)
    sim_state_rec0 = {'t': tf[0], 'x': x0s,  # time, state vector
                      'key': keys0,  # random number generator key
                      'tf': tf,  # overall simulation time frame
                      'check_every_n_steps': int(tau_checkswitchstep / tau),  # tau-leap steps between record points
                      'switch_time_index': jnp.zeros(keys0.shape[0], dtype=jnp.int64),
                      # times of switching, originally set to 0 as a placeholder
                      'switch_time_index1': jnp.zeros(keys0.shape[0], dtype=jnp.int64),
                      # jnp.zeros(keys0.shape[0],dtype=jnp.int64), # times of switching, originally set to 0 as a placeholder
                      'avg_dynamics': False
                      # true if considering the deterministic case with average dynamics of random variables
                      }

    # vmapping - specify that we vmap over the random number generation keys
    sim_state_vmap_axes = {'t': None, 'x': 0,  # time, state vector
                           'key': 0,  # random number generator key
                           'tf': None,  # overall simulation time frame
                           'check_every_n_steps': None,  # tau-leap steps between record points
                           'switch_time_index': 0,  # time of switching individual for each trajectory
                           'switch_time_index1': 0,  # time of switching individual for each trajectory
                           'avg_dynamics': None
                           # true if considering the deterministic cas with average dynamics of random variables
                           }

    # simulate (with vmapping)
    sim_state_rec_final = jax.jit(jax.vmap(tauleap_loop_jit, (sim_state_vmap_axes,)))(sim_state_rec0)

    # get switch times from indices - WILL GIVE TIME SINCE START OF SIMULATION, NOT SINCE T=0!
    switch_times = tau_checkswitchstep * sim_state_rec_final['switch_time_index']
    switch_times1 = tau_checkswitchstep * sim_state_rec_final['switch_time_index1']

    return switch_times, switch_times1, sim_state_rec_final['key']


# check if the next trajectory point is a switching point
def tauleap_check_switch(circuit_v,  # calculating the propensity vector for stochastic simulation of circuit expression
                         circuit_eff_m_het_div_k_het,
                         # calculating the effective mRNA/k values for the synthetic genes in the circuit
                         sim_state_record,  # simulator state
                         i,  # number of steps that is being checked
                         tau,  # time step
                         ode_steps_in_tau,  # number of ODE integration steps in each tau-leap step
                         args,  # arguments for the simulation
                         switch_condition,  # function that returns true if the switching condition is met
                         switch_condition1  # additional function that returns true if the switching condition is met
                         ):
    # DEFINITION OF THE ACTUAL TAU-LEAP SIMULATION STEP
    def tauleap_next_x(step_cntr, sim_state_tauleap):
        # update t
        next_t = sim_state_tauleap['t'] + tau

        # update x
        # find deterministic change in x
        det_update = tauleap_integrate_ode(sim_state_tauleap['t'], sim_state_tauleap['x'], tau, ode_steps_in_tau,
                                           circuit_eff_m_het_div_k_het,
                                           args)
        # find stochastic change in x
        stoch_update = tauleap_update_stochastically(sim_state_tauleap['t'], sim_state_tauleap['x'],
                                                     tau, args, circuit_v,
                                                     sim_state_tauleap['key'], sim_state_tauleap['avg_dynamics'])
        # find next x
        next_x_tentative = sim_state_tauleap['x'] + det_update + stoch_update
        # make sure x has no negative entries
        next_x = jax.lax.select(next_x_tentative < 0, jnp.zeros_like(next_x_tentative), next_x_tentative)
        # next_x=jnp.multiply(next_x_tentative>=0,next_x_tentative)

        # update key
        next_key, _ = jax.random.split(sim_state_tauleap['key'], 2)

        return {
            # entries updated over the course of the tau-leap step
            't': next_t, 'x': next_x,
            'key': next_key,
            # entries unchanged over the course of the tau-leap step
            'tf': sim_state_tauleap['tf'],
            'check_every_n_steps': sim_state_tauleap['check_every_n_steps'],
            'switch_time_index': sim_state_tauleap['switch_time_index'],
            'switch_time_index1': sim_state_tauleap['switch_time_index1'],
            'avg_dynamics': sim_state_tauleap['avg_dynamics']
        }

    # FUNCTION BODY
    # run tau-leap integration until the next state is to be saved
    next_state_bytauleap = jax.lax.fori_loop(0, sim_state_record['check_every_n_steps'], tauleap_next_x,
                                             sim_state_record)

    # check if the switching condition is met, record switching time if yes and not already switched
    switched = switch_condition(sim_state_record['x'], next_state_bytauleap['x'])
    next_state_bytauleap['switch_time_index'] = jax.lax.select(
        jnp.logical_and(switched, sim_state_record['switch_time_index'] == 0),
        i + 1, sim_state_record['switch_time_index'])

    # check the additional switching condition
    switched1 = switch_condition1(sim_state_record['x'], next_state_bytauleap['x'])
    next_state_bytauleap['switch_time_index1'] = jax.lax.select(
        jnp.logical_and(switched1, sim_state_record['switch_time_index1'] == 0),
        i + 1, sim_state_record['switch_time_index1'])

    # update the overall simulator state
    next_sim_state_record = {
        # entries updated over the course of the tau-leap step
        't': next_state_bytauleap['t'], 'x': next_state_bytauleap['x'],
        'key': next_state_bytauleap['key'],
        # entries unchanged
        'tf': sim_state_record['tf'],
        'check_every_n_steps': sim_state_record['check_every_n_steps'],
        'switch_time_index': next_state_bytauleap['switch_time_index'],
        'switch_time_index1': next_state_bytauleap['switch_time_index1'],
        'avg_dynamics': sim_state_record['avg_dynamics']
    }

    return next_sim_state_record


# SIMULATIONS WITH CHANGING CELL VOLUME -------------------------------------------------------------------
def tauleap_sim_varvol(par,  # dictionary with model parameters
                       circuit_v,  # calculating the propensity vector for stochastic simulation of circuit expression
                       circuit_eff_m_het_div_k_het,
                       # calculating the effective mRNA/k values for the synthetic genes in the circuit
                       x0,  # initial condition VECTOR
                       num_circuit_genes, num_circuit_miscs, circuit_name2pos,
                       # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder,
                       sgp4j,  # relevant synthetic gene parameters in jax.array form
                       tf, tau, tau_odestep, tau_savetimestep,
                       # simulation parameters: time frame, tau-leap time step, number of ODE steps in each tau-leap step
                       mRNA_count_scales, S, circuit_synpos2genename, keys0,
                       # parameter vectors for efficient simulation and reaction stoichiometry info - determined by tauleap_sim_prep_varvol!!
                       rep_phase_means_stdevs_bounds,   # params for sampling the randomly distributed replication phase of chr. int. synth. genes - determined by tauleap_sim_prep_varvol!!
                       avg_dynamics=False   # true if considering the deterministic cas with average dynamics of random variables
                       ):
    # define the arguments for finding the next state vector
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        num_circuit_genes, num_circuit_miscs,  # number of genes and miscellaneous species in the circuit
        sgp4j,  # relevant synthetic gene parameters in jax.array form
        mRNA_count_scales, S, circuit_synpos2genename,  # parameters for stochastic simulation
        rep_phase_means_stdevs_bounds
    # parameters for sampling the randomly distributed replication phase of chr. int. synth. genes
    )

    # time points at which we save the solution
    ts = jnp.arange(tf[0], tf[1] + tau_savetimestep / 2, tau_savetimestep)

    # number of ODE steps in each tau-leap step
    ode_steps_in_tau = int(tau / tau_odestep)

    # make the retrieval of next x a lambda-function for jax.lax.scanning
    scan_step = lambda sim_state, t: tauleap_record_x_varvol(circuit_v, circuit_eff_m_het_div_k_het,
                                                             sim_state, t, tau, ode_steps_in_tau,
                                                             args)

    # define the jac.lax.scan function
    tauleap_scan = lambda sim_state_rec0, ts: jax.lax.scan(scan_step, sim_state_rec0, ts)
    tauleap_scan_jit = jax.jit(tauleap_scan)

    # get initial conditions
    if (len(x0.shape) == 1):  # if x0 common for all trajectories, copy it for each trajectory
        x0s = jnp.repeat(x0.reshape((1, x0.shape[0])), keys0.shape[0], axis=0)
    else:  # otherwise, leave initial conditions as is
        x0s = x0

    # get the initial replication volumes for chromosomally integrated synthetic genes
    rep_phases0, key_after_phase_sampling0 = jax.vmap(lambda key: jax.lax.cond(avg_dynamics,
                                                        # if we're considering the deterministic case with average dynamics
                                                        # if considering average dynamics, just return the mean volume at which replication occurs
                                                        lambda rep_phase_means_stdevs_bounds, key: (
                                                        rep_phase_means_stdevs_bounds[0], key),
                                                        # if considering average dynamics, call the random volume sampling function
                                                        tauleap_division_random_phases,
                                                        rep_phase_means_stdevs_bounds, key
                                                        ),in_axes=(0,))(keys0)
    rep_vols0 = (par['V_crit'] / 2) * jnp.exp(rep_phases0)  # calculate the volumes at which replication occurs


    # initalise the simulator state: (t, x, sim_step_cntr, record_step_cntr, key, tf, xs)
    sim_state_rec0 = {'t': tf[0], 'x': x0s,  # time, state vector
                      'key': keys0,  # random number generator key
                      'rep_vols': rep_vols0,  # replication volumes
                      'tf': tf,  # overall simulation time frame
                      'save_every_n_steps': int(tau_savetimestep / tau),  # tau-leap steps between record points
                      'avg_dynamics': avg_dynamics
                      # true if considering the deterministic cas with average dynamics of random variables
                      }

    # vmapping - specify that we vmap over the random number generation keys
    sim_state_vmap_axes = {'t': None, 'x': 0,  # time, state vector
                           'key': 0,  # random number generator key
                           'rep_vols': 0,  # replication volumes
                           'tf': None,  # overall simulation time frame
                           'save_every_n_steps': None,  # tau-leap steps between record points
                           'avg_dynamics': None
                           # true if considering the deterministic cas with average dynamics of random variables
                           }

    # simulate (with vmapping)
    sim_state_rec_final, xs = jax.jit(jax.vmap(tauleap_scan_jit, (sim_state_vmap_axes, None)))(sim_state_rec0, ts)

    return ts, xs, sim_state_rec_final['key']


# log the next trajectory point
def tauleap_record_x_varvol(circuit_v,
                            # calculating the propensity vector for stochastic simulation of circuit expression
                            circuit_eff_m_het_div_k_het,
                            # calculating the effective mRNA/k values for the synthetic genes in the circuit
                            sim_state_record,  # simulator state
                            t,  # time of last record
                            tau,  # time step
                            ode_steps_in_tau,  # number of ODE integration steps in each tau-leap step
                            args):
    # DEFINITION OF THE ACTUAL TAU-LEAP SIMULATION STEP
    def tauleap_next_x(step_cntr, sim_state_tauleap):
        # update t
        next_t = sim_state_tauleap['t'] + tau

        # update x
        # find deterministic change in x
        det_update = tauleap_integrate_ode_varvol(sim_state_tauleap['t'], sim_state_tauleap['x'], tau, ode_steps_in_tau,
                                                  circuit_eff_m_het_div_k_het,
                                                  args)
        # find stochastic change in x
        stoch_update = tauleap_update_stochastically_varvol(sim_state_tauleap['t'], sim_state_tauleap['x'],
                                                            tau, args, circuit_v,
                                                            sim_state_tauleap['rep_vols'],
                                                            sim_state_tauleap['key'], sim_state_tauleap['avg_dynamics'])
        # update key after finding the stochastic update
        key_after_stoch_update, _ = jax.random.split(sim_state_tauleap['key'], 2)

        # find next x, possibly with negative entries
        next_x_possneg = sim_state_tauleap['x'] + det_update + stoch_update
        # make sure x has no negative entries
        next_x_nonneg = jax.lax.select(next_x_possneg < 0, jnp.zeros_like(next_x_possneg), next_x_possneg)

        # if the cell reaches a critical volume, handle cell division with random partitioning of hereologous species
        # note: system parameters is args[0]
        next_x, next_rep_vols, next_key = jax.lax.cond(next_x_nonneg[6] >= args[0]['V_crit'],
                                        # check if cell volume exceeds the critical volume
                                        tauleap_division_varvol,  # if yes, call the division function
                                        lambda x, rep_vols, par, rep_phase_means_stdevs_bounds, key, avg_dynamics: (x, rep_vols, key),
                                        # if no, just return the old state and key
                                        next_x_nonneg, sim_state_tauleap['rep_vols'], args[0], args[8], key_after_stoch_update,
                                        sim_state_tauleap['avg_dynamics'])  # arguments for the division function

        return {
            # entries updated over the course of the tau-leap step
            't': next_t, 'x': next_x,
            'key': next_key,
            'rep_vols': next_rep_vols,
            # entries unchanged over the course of the tau-leap step
            'tf': sim_state_tauleap['tf'],
            'save_every_n_steps': sim_state_tauleap['save_every_n_steps'],
            'avg_dynamics': sim_state_tauleap['avg_dynamics']
        }

    # FUNCTION BODY
    # run tau-leap integration until the next state is to be saved
    next_state_bytauleap = jax.lax.fori_loop(0, sim_state_record['save_every_n_steps'], tauleap_next_x,
                                             sim_state_record)

    # update the overall simulator state
    next_sim_state_record = {
        # entries updated over the course of the tau-leap step
        't': next_state_bytauleap['t'], 'x': next_state_bytauleap['x'],
        'key': next_state_bytauleap['key'],
        'rep_vols': next_state_bytauleap['rep_vols'],
        # entries unchanged
        'tf': sim_state_record['tf'],
        'save_every_n_steps': sim_state_record['save_every_n_steps'],
        'avg_dynamics': sim_state_record['avg_dynamics']
    }

    return next_sim_state_record, sim_state_record['x']


# cell division
def tauleap_division_varvol(x, rep_vols,
                            par, rep_phase_means_stdevs_bounds,
                            key, avg_dynamics):
    # get x with partitioning of species
    x_with_partitioning, key_after_partitioning = jax.lax.cond(avg_dynamics,    # if we're considering the deterministic case with average dynamics
                                                               # if considering average dynamics, just divide all abundances (as well as volume at x[6]) by two
                                                               lambda x, key: (x / 2, key),
                                                               # if considering average dynamics, call the random partitioning function
                                                               tauleap_division_varvol_random_partition,
                                                               x, key
                                                               )
    # get cellcycle phases at which chromosomally integrated synthetic genes will replicate
    rep_phases, key_after_phase_sampling = jax.lax.cond(avg_dynamics,   # if we're considering the deterministic case with average dynamics
                                                    # if considering average dynamics, just return the mean volume at which replication occurs
                                                    lambda rep_phase_means_stdevs_bounds, key: (rep_phase_means_stdevs_bounds[0], key),
                                                    # if considering average dynamics, call the random volume sampling function
                                                    tauleap_division_random_phases,
                                                    rep_phase_means_stdevs_bounds, key_after_partitioning
                                                    )
    rep_vols=(par['V_crit']/2)*jnp.exp(rep_phases)  # calculate the volumes at which replication occurs

    # return the state vector with new cell volume and partitioned heterologous species, as well as the new random number generation key
    return x_with_partitioning, rep_vols, key_after_phase_sampling

# at time of cell division, partition molecules of species between two daughter cells
def tauleap_division_varvol_random_partition(x, key):
    # sample the heterologous species' abundances from a binomial distribution with equal probability of partitioning to each daughter cell
    x_with_partitioning = jnp.concatenate((x[0:8]/2, jax.random.binomial(key, x[8:], 0.5)))

    #  update the key
    key_after_partitioning,_ = jax.random.split(key, 2)

    return x_with_partitioning, key_after_partitioning


# at time of cell division, sample the cell cycle phases at which replication of chr. int. synth. genes occurs
def tauleap_division_random_phases(rep_phase_means_stdevs_bounds, key):
    # sample the phases at which replication occurs
    rep_phases = jax.random.truncated_normal(key,  # random number generator key
                                             rep_phase_means_stdevs_bounds[2],  # lower bound
                                             rep_phase_means_stdevs_bounds[3]  # upper bound
                                             ) * rep_phase_means_stdevs_bounds[1] + rep_phase_means_stdevs_bounds[0]

    # if random sampling has been used, update the key
    key_after_volume_sampling, _ = jax.random.split(key, 2)

    return rep_phases, key_after_volume_sampling


# ode integration - Euler method
def tauleap_integrate_ode_varvol(t, x, tau, ode_steps_in_tau,
                                 circuit_eff_m_het_div_k_het,
                                 args):
    return tauleap_ode_varvol(t, x, circuit_eff_m_het_div_k_het, args) * tau


# ode for the deterministic part of the tau-leaping simulation
def tauleap_ode_varvol(t, x, circuit_eff_m_het_div_k_het, args):
    # unpack the args
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]  # number of genes in the circuit
    num_circuit_miscs = args[3]  # number of miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het, g_het = args[4]  # unpack jax-arrayed synthetic gene parameters

    # GET THE STATE VECTOR WITH SPECIE CONCENTRATIONS (divide them by the cell volume)
    V = x[6]
    x_concs = x / V

    # give the state vector entries meaningful names
    m_a = x_concs[0]  # metabolic gene mRNA
    m_r = x_concs[1]  # ribosomal gene mRNA
    p_a = x_concs[2]  # metabolic proteins
    R = x_concs[3]  # non-inactivated ribosomes
    tc = x_concs[4]  # charged tRNAs
    tu = x_concs[5]  # uncharged tRNAs
    s = par['s']  # CAREFUL: NUTRIENT QUALITY S IS A PARAMETER, ASSUMED CONSTANT
    h = x_concs[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with circuit_name2pos

    # FIND SPECIAL SYNTHETIC PROTEIN CONCENTRATIONS - IF PRESENT
    # chloramphenicol acetyltransferase (antibiotic reistance)
    p_cat = jax.lax.select(par['cat_gene_present'] == 1, x_concs[circuit_name2pos['p_cat']], 0.0)
    # synthetic protease (synthetic protein degradation)
    p_prot = jax.lax.select(par['prot_gene_present'] == 1, x_concs[circuit_name2pos['p_prot']], 0.0)

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])
    k_het = k_calc(e, kplus_het, kminus_het, n_het)

    # ratio of charged to uncharged tRNAs
    T = tc / tu

    # corection to ribosome availability due to chloramphenicol action
    H = (par['K_D'] + h) / par['K_D']

    # heterologous mRNA levels scaled by RBS strength
    m_het_div_k_het = circuit_eff_m_het_div_k_het(x_concs, par, circuit_name2pos, num_circuit_genes, e, k_het)

    # heterologous protein degradation flux
    prodeflux = jnp.sum(
        # (degradation rate times protease level times protein concnetration) times number of AAs per protein
        d_het * p_prot * x_concs[8 + num_circuit_genes:8 + num_circuit_genes * 2] * n_het
    )
    prodeflux_times_H_div_eR = prodeflux * H / (e * R)  # degradation flux scaled by overall protein synthesis rate

    # resource competition denominator
    m_notq_div_k_notq = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_times_H_div_eR) * m_notq_div_k_notq - par[
        'phi_q'] * prodeflux_times_H_div_eR) / \
                (1 - par['phi_q'] * (1 - prodeflux_times_H_div_eR))
    D = H * (1 + mq_div_kq + m_notq_div_k_notq)
    B = R * (1 / H - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    metab_burd = jnp.sum(g_het * x_concs[
                                 8 + num_circuit_genes:8 + num_circuit_genes * 2])  # metabolic burden imposed by synthetic proteins

    nu = nu_calc(par, tu, s, metab_burd)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # continuously, we only consider charged aa-tRNA consumption by NATIVE protein translation
    B_cont = R * (1 / H - 1 / D - jnp.sum(jnp.divide(x_concs[8:8 + num_circuit_genes], k_het)) / D)

    # return dx/dt for the host cell; CAREFUL: NO DILUTION TERM, AND dx/dt per 1um^3 IS MULTIPLED BY V
    return V * jnp.array([
                             # mRNAs
                             l * par['c_a'] * par['a_a'] - par['b_a'] * m_a,
                             l * Fr_calc(par, T) * par['c_r'] * par['a_r'] - par['b_r'] * m_r,
                             # metabolic protein p_a
                             (e / par['n_a']) * (m_a / k_a / D) * R,
                             # ribosomes
                             (e / par['n_r']) * (m_r / k_r / D) * R,
                             # tRNAs
                             nu * p_a - e * B_cont,
                             l * psi - nu * p_a + e * B_cont,
                             # volume increases at growth rate (mind that we multiply by V before jnp.array())
                             l,
                             # chloramphenicol concentration
                             par['diff_h'] * (par['h_ext'] - h) - h * p_cat / par['K_C']
                         ] +
                         [0] * (2 * num_circuit_genes) + [0] * num_circuit_miscs
                         # synthetic gene expression considered stochastically
                         )


# stochastic update calculation
def tauleap_update_stochastically_varvol(t, x, tau, args, circuit_v,
                                         rep_vols,  # volumes at which chromosomally integrated synthetic genes replicate
                                         key, avg_dynamics):
    # PREPARATION
    # unpack the arguments
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]  # number of genes in the circuit
    num_circuit_miscs = args[3]  # number of miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het, g_het = args[4]  # unpack jax-arrayed synthetic gene parameters
    # stochastic simulation arguments
    mRNA_count_scales = args[5]
    S = args[6]
    circuit_synpos2genename = args[7]  # parameter vectors for efficient simulation and reaction stoichiometry info

    # GET THE STATE VECTOR WITH SPECIE CONCENTRATIONS (divide them by the cell volume)
    V = x[6]
    x_concs = x / V

    # give the state vector entries meaningful names
    m_a = x_concs[0]  # metabolic gene mRNA
    m_r = x_concs[1]  # ribosomal gene mRNA
    p_a = x_concs[2]  # metabolic proteins
    R = x_concs[3]  # non-inactivated ribosomes
    tc = x_concs[4]  # charged tRNAs
    tu = x_concs[5]  # uncharged tRNAs
    s = par['s']  # CAREFUL: NUTRIENT QUALITY S IS A PARAMETER, ASSUMED CONSTANT
    h = x_concs[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x_concs with circuit_name2pos

    # FIND SPECIAL SYNTHETIC PROTEIN CONCENTRATIONS - IF PRESENT
    # chloramphenicol acetyltransferase (antibiotic reistance)
    p_cat = jax.lax.select(par['cat_gene_present'] == 1, x_concs[circuit_name2pos['p_cat']], 0.0)
    # synthetic protease (synthetic protein degradation)
    p_prot = jax.lax.select(par['prot_gene_present'] == 1, x_concs[circuit_name2pos['p_prot']], 0.0)

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])
    k_het = k_calc(e, kplus_het, kminus_het, n_het)

    T = tc / tu  # ratio of charged to uncharged tRNAs

    H = (par['K_D'] + h) / par['K_D']  # corection to ribosome availability due to chloramphenicol action

    m_het_div_k_het = jnp.sum(x_concs[8:8 + num_circuit_genes] / k_het)  # heterologous protein synthesis flux
    prodeflux = jnp.sum(
        d_het * n_het * x_concs[
                        8 + num_circuit_genes:8 + num_circuit_genes * 2])  # heterologous protein degradation flux
    prodeflux_times_H_div_eR = prodeflux * H / (e * R)

    # resource competition denominator
    m_notq_div_k_notq = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_times_H_div_eR) * m_notq_div_k_notq - par[
        'phi_q'] * prodeflux_times_H_div_eR) / \
                (1 - par['phi_q'] * (1 - prodeflux_times_H_div_eR))
    D = H * (1 + mq_div_kq + m_notq_div_k_notq)
    B = R * (1 / H - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    metab_burd = jnp.sum(g_het * x_concs[
                                 8 + num_circuit_genes:8 + num_circuit_genes * 2])  # total metabolic burden imposed by synthetic proteins

    nu = nu_calc(par, tu, s, metab_burd)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # continuously, we only consider charged aa-tRNA consumption by NATIVE protein translation
    B_cont = (D - 1 - jnp.sum(jnp.divide(x_concs[8:8 + num_circuit_genes], k_het))) / D * R

    # FIND REACTION PROPENSITIES
    # per 1 um^3 of cell volume
    v_per_vol = jnp.array(circuit_v(t, x_concs,  # time, cell state, external inputs
                                    e, l,  # translation elongation rate, growth rate
                                    R,  # ribosome count in the cell
                                    k_het, D,
                                    # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
                                    p_prot,  # synthetic protease concentration
                                    mRNA_count_scales,  # scaling factors for synthetic gene mRNA counts
                                    par,  # system parameters
                                    circuit_name2pos,    # gene name - position in circuit vector decoder
                                    V, # cell volume
                                    rep_vols    # volumes at which chromosomally integrated synthetic genes replicate
                                    ))
    # in the entire cell volume
    v = v_per_vol * V

    # RETURN THE NUMBER OF TIMES THAT EACH REACTION HAS OCCURRED
    # or take the average if we're considering the deterministic case
    return jax.lax.select(avg_dynamics, jnp.matmul(S, v * tau), jnp.matmul(S, jax.random.poisson(key=key, lam=v * tau)))


# preparatory step creating the objects necessary for tau-leap simulation - DO NOT JIT
def tauleap_sim_prep_varvol(par,  # dictionary with model parameters
                            num_circuit_genes, num_circuit_miscs, circuit_name2pos,
                            x0_det,  # initial condition found deterministically
                            key_seeds=[0]  # random key seed(s)
                            ):
    # DICTIONARY mapping positions of a synthetic gene in the list of all synthetic genes to their names
    circuit_synpos2genename = {}
    for name in circuit_name2pos.keys():
        if (name[0] == 'm'):
            circuit_synpos2genename[circuit_name2pos[name] - 8] = name[2:]

    # PARAMETER VECTORS FOR EFFICIENT SIMULATION
    # mRNA count scalings
    mRNA_count_scales_np = np.zeros((num_circuit_genes))
    for i in range(0, num_circuit_genes):
        mRNA_count_scales_np[i] = par['n_' + circuit_synpos2genename[i]] / 25
    mRNA_count_scales = jnp.array(mRNA_count_scales_np)

    # STOICHIOMETRY MATRIX - NOTE THAT A DIFFERENT ONE FROM THE BASIC TAU-LEAPING SIMULATION IS USED
    S = gen_stoich_mat_varvol(par,
                              circuit_name2pos, circuit_synpos2genename,
                              num_circuit_genes, num_circuit_miscs,
                              mRNA_count_scales)

    # MAKE THE INITIAL CONDITION FOR TAU-LEAPING WITH APPROPRIATE INTEGER VALUES OF RANDOM VARIABLES
    x0_det_np = np.array(x0_det)  # convert to numpy array for convenience
    x0_tauleap_mrnas = np.multiply(np.round(x0_det_np[8:8 + num_circuit_genes] / mRNA_count_scales_np),
                                   mRNA_count_scales_np)  # synthetic mRNA counts - must be a multiple of the corresponding scaling factors that acount for translation by multiple ribosomes
    x0_tauleap_prots_and_miscs = np.round(
        x0_det_np[8 + num_circuit_genes:])  # synthetic protein and miscellaneous species counts
    x0_tauleap = np.concatenate((x0_det_np[0:8], x0_tauleap_mrnas, x0_tauleap_prots_and_miscs), axis=0)

    # STARTING RANDOM NUMBER GENERATION KEY
    key_seeds_jnp = jnp.array(key_seeds)
    keys0 = jax.vmap(jax.random.PRNGKey)(key_seeds_jnp)

    # RANDOMLY SAMPLED REPLICATION PHASES FOR CHROMOSOMALLY INTEGRATED SYNTHETIC GENES
    # initialise lists of mean, st dev and bounds for sampling the replication phase
    rep_phase_mean = []
    rep_phase_stdev = []
    rep_phase_bound_low = []
    rep_phase_bound_high = []
    for i in range(0, num_circuit_genes):
        # if mean replication phase stated as -1, means the gene is NOT chromosomally integrated
        if (par['mean_rep_phase_' + circuit_synpos2genename[i]] > 0):
            rep_phase_mean.append(par['mean_rep_phase_' + circuit_synpos2genename[i]])
            rep_phase_stdev.append(par['stdev_rep_phase_' + circuit_synpos2genename[i]])
            rep_phase_bound_low.append((0 - par['mean_rep_phase_' + circuit_synpos2genename[i]]) / par[
                'stdev_rep_phase_' + circuit_synpos2genename[i]])
            rep_phase_bound_high.append((1 - par['mean_rep_phase_' + circuit_synpos2genename[i]]) / par[
                'stdev_rep_phase_' + circuit_synpos2genename[i]])
    # gather all the records together
    rep_phase_means_stdevs_bounds = jnp.array([rep_phase_mean,
                                               rep_phase_stdev,
                                               rep_phase_bound_low,
                                               rep_phase_bound_high])

    return mRNA_count_scales, S, x0_tauleap, circuit_synpos2genename, keys0, rep_phase_means_stdevs_bounds


# generate the stocihiometry matrix for heterologous genes (plus return total number of stochastic reactions) - DO NOT JIT
# WITH VARIABLE CELL VOLUME, DOES NOT HAVE DILUTION TERMS!
def gen_stoich_mat_varvol(par,
                          circuit_name2pos, circuit_synpos2genename,
                          num_circuit_genes, num_circuit_miscs,
                          mRNA_count_scales
                          ):
    # unpack stochionmetry args
    mRNA_count_scales_np = np.array(mRNA_count_scales)

    # find the number of stochastic reactions that can occur
    num_stoch_reactions = num_circuit_genes * (2 +  # synthesis/degradation/dilution of mRNA
                                               2)  # synthesis/degradation/dilution of protein
    if ('cat_pb' in circuit_name2pos.keys()):  # plus, might need to model stochastic action of integrase
        num_stoch_reactions += 2  # functional CAT gene forward and reverse strain exchange, LR site-integrase dissociation due to conformation change and plasmid replication
        num_stoch_reactions += 1  # LR site-integrase dissociation due to conformation change
        num_stoch_reactions += 3  # replication of the plasmid with LR site-integrase complex, functional CAT gene or no CAT gene
        num_stoch_reactions += 2  # synthesis and degradation of the replication inhibitor

    # initialise (in numpy format)
    S = np.zeros((8 + 2 * num_circuit_genes + num_circuit_miscs, num_stoch_reactions))

    # initialise thge counter of reactions in S
    reaction_cntr = 0

    # mRNA - reactions common for all genes
    for i in range(0, num_circuit_genes):
        S[8 + i, reaction_cntr] = mRNA_count_scales_np[i]  # mRNA synthesis
        reaction_cntr += 1
        S[8 + i, reaction_cntr] = -mRNA_count_scales_np[i]  # mRNA degradation
        reaction_cntr += 1

    # protein - reactions common for all genes
    for i in range(0, num_circuit_genes):
        S[8 + num_circuit_genes + i, reaction_cntr] = 1  # protein synthesis
        S[4, reaction_cntr] = -par[
            'n_' + circuit_synpos2genename[i]]  # includes tRNA unchraging during translation(-tc)
        S[5, reaction_cntr] = par['n_' + circuit_synpos2genename[i]]  # includes tRNA unchraging during translation(+tu)
        reaction_cntr += 1
        S[8 + num_circuit_genes + i, reaction_cntr] = -1  # protein degradation
        reaction_cntr += 1

    if ('cat_pb' in circuit_name2pos.keys()):  # plasmid dynamics, including stochastic action of integrase
        # CAT gene forward strain exchange: from cat_pb to cat_lri1
        S[8 + 2 * num_circuit_genes, reaction_cntr] = -1  # cat_pb decreased
        S[8 + 2 * num_circuit_genes + 1, reaction_cntr] = 1  # cat_lri1 increased
        reaction_cntr += 1

        # CAT gene reverse strain exchange: from cat_lri1 to cat_pb
        S[8 + 2 * num_circuit_genes, reaction_cntr] = 1  # cat_pb increased
        S[8 + 2 * num_circuit_genes + 1, reaction_cntr] = -1  # cat_lri1 decreased
        reaction_cntr += 1

        # LR site-integrase dissociation due to conformation change
        S[8 + 2 * num_circuit_genes + 1, reaction_cntr] = -1  # cat_lri1 decreased
        reaction_cntr += 1

        # LR site-integrase dissociation due to plasmid replication
        S[8 + 2 * num_circuit_genes + 1, reaction_cntr] = -1  # cat_lri1 decreased
        S[
            8 + 2 * num_circuit_genes + 2, reaction_cntr] = 2  # both the newly replicated plasmid and the old one are now CAt-less
        reaction_cntr += 1

        # replication of the plasmid with a functional CAT gene
        S[8 + 2 * num_circuit_genes, reaction_cntr] = 1
        reaction_cntr += 1

        # replication of the CAT-less plasmid
        S[8 + 2 * num_circuit_genes + 2, reaction_cntr] = 1
        reaction_cntr += 1

        # synthesis of the replication inhibitor
        S[8 + 2 * num_circuit_genes + 3, reaction_cntr] = 1
        reaction_cntr += 1

        # degradation of the replication inhibitor
        S[8 + 2 * num_circuit_genes + 3, reaction_cntr] = -1

    return jnp.array(S)


# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    # initialise cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions

    # load synthetic gene circuit - WITH HYBRID SIMULATION SUPPORT
    ode_with_circuit, circuit_F_calc, circuit_eff_m_het_div_k_het, \
        par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles, circuit_v = cellmodel_auxil.add_circuit(
        circuits.punisher_cnc_b_initialise,
        circuits.punisher_cnc_b_ode,
        circuits.punisher_cnc_b_F_calc,
        circuits.punisher_cnc_b_eff_m_het_div_k_het,
        par, init_conds,
        # propensity calculation function for hybrid simulations
        circuits.punisher_cnc_b_v_varvol, varvol=True)

    # BURDENSOME SYNTHETIC GENE
    par['c_b'] = 1
    par['a_b'] = 1e5

    # PUNISHER
    # switch gene conc
    par['a_switch'] = 400.0  # promoter strength (unitless)
    par['d_switch'] = 0.01836
    # integrase - expressed from the switch gene's operon, not its own gene => c_int, a_int irrelevant
    par['k+_int'] = par['k+_switch'] / 80.0  # RBS weaker than for the switch gene
    par['d_int'] = 0.0  # 0.01836 # rate of integrase degradation per protease molecule (1/nM/h)
    # CAT (antibiotic resistance) gene
    par['a_cat'] = 500.0  # promoter strength (unitless)
    par['n_cat'] = 300.0
    # synthetic protease gene
    par['a_prot'] = 25.0  # promoter strength (unitless)
    init_conds['p_prot'] = 1500.0  # if zero at start, the punisher's triggered prematurely

    # punisher's transcription regulation function
    par['K_switch'] = 300.0  # Half-saturation constant for the self-activating switch gene promoter (nM)
    par['eta_switch'] = 2  # Hill coefficient for the self-activating switch gene promoter (unitless)
    par['baseline_switch'] = 0.025  # Baseline value of the switch gene's transcription activation function
    par['p_switch_ac_frac'] = 0.85  # active fraction of protein (i.e. share of molecules bound by the inducer)

    # plasmid copy number control
    init_conds['cat_pb'] = 10.0  # INITIAL CONDITION (not a parameter): all plasmids have working CAT gene copies
    par['k_tr'] = 130.2  # plasmid replication rate (1/h)
    par['a_inh'] = 948  # inhibitor synthesis rate per plasmid copy (1/h)
    par['b_inh'] = 74.976  # inhibitor degradation rate (1/h)
    par['n_inh'] = 10  # number of steps of replication initiation at which inhibition can happen
    par['K_inh'] = 214.05  # replication inhibition constant (nM)

    # critical cell volume triggering division
    par['V_crit'] = 2.0 * np.log(2)  # 2ln(2) so as to have an average volume of 1 um^3 assuming constant growth rate

    # burdensome gene replication
    par['mean_rep_phase_b'] = 0.5  # mean replication phase
    par['stdev_rep_phase_b'] = 0.23  # standard deviation of replication phase (void as considering avergae dynamics for now)
    c_b_scale = 1#(par['V_crit']/2) * np.log(2) * np.exp(par['mean_rep_phase_b']*np.log(2))
    print(c_b_scale)

    # culture medium
    nutr_qual = 0.5
    par['s'] = nutr_qual  # nutrient quality (unitless)
    init_conds['s'] = nutr_qual  # nutrient quality (unitless)

    # DETERMINISTIC SIMULATION TO FIND THE STARTING STEADY STATE
    # set simulation parameters
    tf = (0, 50)  # simulation time frame
    savetimestep = 0.1  # save time step
    dt_max = 0.1  # maximum integration step
    rtol = 1e-6  # relative tolerance for the ODE solver
    atol = 1e-6  # absolute tolerance for the ODE solver

    # simulate
    sol = ode_sim(par,  # dictionary with model parameters
                  ode_with_circuit,  # ODE function for the cell with synthetic circuit
                  cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                  # initial condition VECTOR
                  len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                  cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                  # synthetic gene parameters for calculating k values
                  tf, jnp.arange(tf[0], tf[1], savetimestep),  # time axis for saving the system's state
                  rtol,
                  atol)  # simulation parameters: when to save the system's state, relative and absolute tolerances)   # simulation parameters: time frame, save time step, relative and absolute tolerances
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)
    # det_steady_x = jnp.concatenate((sol.ys[-1, 0:8], jnp.round(sol.ys[-1, 8:])))
    det_steady_x = sol.ys[-1, :]

    # HYBRID SIMULATION WITH VARIABLE CELL VOLUME
    # tau-leap hybrid simulation parameters
    tf_hybrid = (tf[-1], tf[-1] + 0.6)  # simulation time frame
    tau = 1e-7  # simulation time step
    tau_odestep = 1e-7  # number of ODE integration steps in a single tau-leap step (smaller than tau)
    tau_savetimestep = 2e-2  # save time step a multiple of tau

    # simulate
    timer = time.time()
    mRNA_count_scales, S, x0_tauleap, circuit_synpos2genename, keys0, \
        rep_phase_means_stdevs_bounds = tauleap_sim_prep_varvol(par, len(circuit_genes),
                                                                len(circuit_miscs),
                                                                circuit_name2pos,
                                                                det_steady_x,
                                                                key_seeds=[0]
                                                                )
    x0_tauleap[6] = 1.0  # start at the default volume of 1 um^3
    par['a_b']=par['a_b']*c_b_scale
    ts_jnp, xs_jnp, final_keys = tauleap_sim_varvol(par,  # dictionary with model parameters
                                                    circuit_v,  # circuit reaction propensity calculator
                                                    circuit_eff_m_het_div_k_het,
                                                    x0_tauleap,
                                                    # initial condition VECTOR (processed to make sure random variables are appropriate integers)
                                                    len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                                                    cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                                                    # synthetic gene parameters for calculating k values
                                                    tf_hybrid, tau, tau_odestep, tau_savetimestep,
                                                    # simulation parameters: time frame, tau leap step size, number of ode integration steps in a single tau leap step
                                                    mRNA_count_scales, S, circuit_synpos2genename,
                                                    # mRNA count scaling factor, stoichiometry matrix, synthetic gene number in list of synth. genes to name decoder
                                                    keys0, # starting random number genereation keys
                                                    rep_phase_means_stdevs_bounds,  # params for sampling the randomly distributed replication phase of chr. int. synth. genes - determined by tauleap_sim_prep_varvol!!
                                                    avg_dynamics=False)

    # concatenate the results with the deterministic simulation
    ts = np.concatenate((ts, np.array(ts_jnp)))
    xs_first = np.concatenate(
        (xs, np.array(xs_jnp[0])))  # getting the results from the first random number generator key in vmap
    xss = np.concatenate((xs * np.ones((keys0.shape[0], 1, 1)), np.array(xs_jnp)),
                         axis=1)  # getting the results from all vmapped trajectories

    print('tau-leap simulation time: ', time.time() - timer)

    # np.save('xs_first_fluct_varvol.npy', xs_first)
    # np.save('ts_fluct_varvol.npy', ts)
    # np.save('xss_fluct_varvol.npy', xss)

    # xs_first=np.load('xs_first_fluct_varvol15.npy')
    # ts=np.load('ts_fluct_varvol15.npy')
    # xss=np.load('xss_fluct_varvol15.npy')

    # PLOT - HOST CELL MODEL
    bkplot.output_file(filename="cellmodel_sim.html",
                       title="Cell Model Simulation")  # set up bokeh output file
    mass_fig = cellmodel_auxil.plot_protein_masses(ts, xs_first, par, circuit_genes)  # plot simulation results
    nat_mrna_fig, nat_prot_fig, nat_trna_fig, h_fig = cellmodel_auxil.plot_native_concentrations(ts, xs_first, par,
                                                                                                 circuit_genes,
                                                                                                 tspan=tf_hybrid,
                                                                                                 varvol=True)  # plot simulation results
    l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure = cellmodel_auxil.plot_phys_variables(ts,
                                                                                                           xs_first,
                                                                                                           par,
                                                                                                           circuit_genes,
                                                                                                           circuit_miscs,
                                                                                                           circuit_name2pos,
                                                                                                           circuit_eff_m_het_div_k_het,
                                                                                                           tspan=tf_hybrid,
                                                                                                           varvol=True)  # plot simulation results
    vol_figure = cellmodel_auxil.plot_volume(ts, xs_first, par, circuit_genes,
                                             tspan=tf_hybrid)  # plot simulation results
    bkplot.save(bklayouts.grid([[nat_mrna_fig, nat_prot_fig, vol_figure],
                                [nat_trna_fig, h_fig, l_figure],
                                [e_figure, Fr_figure, D_figure]]))

    # PLOT - SYNTHETIC GENE CIRCUIT
    bkplot.output_file(filename="circuit_sim.html",
                       title="Synthetic Gene Circuit Simulation")  # set up bokeh output file
    het_mrna_fig, het_prot_fig, misc_fig = cellmodel_auxil.plot_circuit_concentrations(ts, xs_first, par,
                                                                                       circuit_genes,
                                                                                       circuit_miscs,
                                                                                       circuit_name2pos,
                                                                                       circuit_styles, tspan=tf_hybrid,
                                                                                       varvol=False)  # plot simulation results
    F_fig = cellmodel_auxil.plot_circuit_regulation(ts, xs_first, circuit_F_calc, par, circuit_genes, circuit_miscs,
                                                    circuit_name2pos, circuit_styles, tspan=tf_hybrid,
                                                    varvol=True)  # plot simulation results
    bkplot.save(bklayouts.grid([[het_mrna_fig, het_prot_fig, misc_fig],
                                [F_fig, None, None]]))

    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
