'''
SYNTHETIC_CIRCUITS_JAX.PY: Describing different synthetic gene circuits
for the Python/Jax implementation of the coarse-grained resource-aware E.coli model
'''


# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

# NO SYNTHETIC GENES ---------------------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def nocircuit_initialise():
    default_par={'cat_gene_present':0, 'prot_gene_present':0} # chloramphenicol resistance or synthetic protease gene not present, natrually
    default_init_conds={}
    genes={}
    miscs={}
    name2pos={'p_cat':0, } # placeholders, will never be used but required for correct execution'}
    circuit_colours={}
    return default_par, default_init_conds, genes, miscs, name2pos, circuit_colours

# transcription regulation functions
def nocircuit_F_calc(t ,x, par, name2pos):
    return jnp.array([])

# ode
def nocircuit_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # RETURN THE ODE
    return []


# stochastic reaction propensities for hybrid tau-leaping simulations
def nocircuit_v(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            mRNA_count_scales, # scaling factors for mRNA counts
            par,  # system parameters
            name2pos
            ):
    # RETURN THE PROPENSITIES
    return []

# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def nocircuit_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                  e, k_het):
    return 0


# ONE CONSTITUTIVE GENE [tau-leap compatible]---------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def oneconstitutive_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['b']  # names of genes in the circuit
    miscs = []  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] =  i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat']=0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot']=0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def oneconstitutive_F_calc(t ,x, par, name2pos):
    F_b = 1 # constitutive gene
    return jnp.array([F_b])

# ode
def oneconstitutive_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] - (par['b_b'] + l) * x[name2pos['m_b']],
            # proteins
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R - (l + par['d_b']*p_prot) * x[name2pos['p_b']]
    ]

# stochastic reaction propensities for hybrid tau-leaping simulations
def oneconstitutive_v(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            mRNA_count_scales, # scaling factors for mRNA counts
            par,  # system parameters
            name2pos
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE PROPENSITIES
    return [
            # synthesis, degradation, dilution of burdensome synthetic gene mRNA
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] / mRNA_count_scales[name2pos['mscale_b']],
            par['b_b'] * x[name2pos['m_b']] / mRNA_count_scales[name2pos['mscale_b']],
            l * x[name2pos['m_b']] / mRNA_count_scales[name2pos['mscale_b']],
            # synthesis, degradation, dilution of burdensome synthetic gene protein
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R,
            par['d_b'] * p_prot * x[name2pos['p_b']],
            l * x[name2pos['p_b']]
    ]

# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def oneconstitutive_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                  e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, no correction as no co-expression
    return m_het_div_k_het_uncorr

# ONE CONSTITUTIVE GENE + CHLORAMPHENICOL RESISTANCE -------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
def oneconstitutive_cat_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['b','cat']  # names of genes in the circuit
    miscs = []  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles={'colours':{}, 'dashes':{}} # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def oneconstitutive_cat_F_calc(t ,x, par, name2pos):
    F_b = 1 # constitutive gene
    F_cat = 1 # constitutive gene
    return jnp.array([F_b, F_cat])

# ode
def oneconstitutive_cat_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] - (par['b_b'] + l) * x[name2pos['m_b']],
            par['func_cat'] * l * F[name2pos['F_cat']] * par['c_cat'] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],
            # proteins
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R - (l + par['d_b']*p_prot) * x[name2pos['p_b']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']*p_prot) * x[name2pos['p_cat']]
    ]


# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def oneconstitutive_cat_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                        e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, no correction as no co-expression
    return m_het_div_k_het_uncorr

# ONE CONSTITUTIVE GENE + CHLORAMPHENICOL RESISTANCE + SYNTHETIC PROTEASE ----------------------------------------------
# initialise all the necessary parameters to simulate the circuit
def oneconstitutive_cat_prot_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['b','cat','prot']  # names of genes in the circuit
    miscs = []  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles={'colours':{}, 'dashes':{}} # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def oneconstitutive_cat_prot_F_calc(t ,x, par, name2pos):
    F_b = 1 # constitutive gene
    F_cat = 1 # constitutive gene
    F_prot = 1 # constitutive gene
    return jnp.array([F_b, F_cat, F_prot])

# ode
def oneconstitutive_cat_prot_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] - (par['b_b'] + l) * x[name2pos['m_b']],
            par['func_cat'] * l * F[name2pos['F_cat']] * par['c_cat'] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],
            par['func_prot'] * l * F[name2pos['F_prot']] * par['c_prot'] * par['a_prot'] - (par['b_prot'] + l) * x[name2pos['m_prot']],
            # proteins
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R - (l + par['d_b']*p_prot) * x[name2pos['p_b']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']*p_prot) * x[name2pos['p_cat']],
            (e / par['n_prot']) * (x[name2pos['m_prot']] / k_het[name2pos['k_prot']] / D) * R - (l + par['d_prot']*p_prot) * x[name2pos['p_prot']]
    ]


# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def oneconstitutive_cat_prot_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                        e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, no correction as no co-expression
    return m_het_div_k_het_uncorr


# PUNISHER AND A SINGLE BURDENSOME GENE [tau-leap compatible, includes a synthetic protease]----------------------------
def punisher_b_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['b',
             'switch', 'int', 'cat', 'prot']  # names of genes in the circuit
    miscs = ['cat_pb', 'cat_lri1']  # names of miscellaneous species involved in the circuit. Here, cat gene states with respect to the integrase
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # punisher
    default_par['c_switch'] = 1  # gene concentration (nM)
    default_par['a_switch'] = 150  # promoter strength (unitless)
    default_par['c_int'] = 0  # integrase expressed from the same operon as the switch protein
    default_par['a_int'] = 0 # integrase expressed from the same operon as the switch protein
    default_par['c_cat'] = 1  # gene concentration (nM)
    default_par['a_cat'] = 5000  # promoter strength (unitless)

    # transcription regulation function
    default_par['K_switch'] = 100
    default_par['eta_switch'] = 2
    default_par['baseline_switch'] = 0.1
    default_par['p_switch_ac_frac'] = 1  # ACTIVE fraction of protein (i.e. share of molecules bound by the inducer)

    # integrase action
    default_par['K_bI~'] = 100 # dissociation constant for the integrase and DNA (nM)
    default_par['k_sxf'] = 6 # forward strain exchange rate by the integrase(1/h)
    default_par['k_sxr'] = 2.14 # reverse strain exchange rate by the integrase(1/h)
    default_par['k_+syn'] = 0.006 # forward synaptic conformational change rate (1/h)

    # initial conditions for the state of cat gene with respect to the integrase
    default_init_conds['cat_pb'] = 10 # originally, the gene is in functional state (with attP and attB sites)
    default_init_conds['cat_lri1'] = 0 # originally, the gene is in functional state (with attP and attB sites)
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles={'colours':{}, 'dashes':{}} # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # give cat gene states same colours as cat mRNA and protein but different dashes
    circuit_styles['colours']['cat_pb'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['cat_lri1'] = circuit_styles['colours']['cat']

    circuit_styles['dashes']['cat_pb'] = 'solid' # fully functional gene => solid dash
    circuit_styles['dashes']['cat_lri1'] = 'dashed' # non-functional => non-solid
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def punisher_b_F_calc(t ,x, par, name2pos):
    F_b = 1 # constitutive gene
    F_cat = 1 # constitutive gene

    # switch and integrase genes regulated by p_switch
    p_switch_dependent_term = (x[name2pos['p_switch']]*par['p_switch_ac_frac']/par['K_switch'])**par['eta_switch']
    F_switch = par['baseline_switch'] + (1 - par['baseline_switch']) * (p_switch_dependent_term/(p_switch_dependent_term+1))
    F_int = 0    # integrase expressed from the same operon as the switch protein, not its own gene
    F_prot = 1 # constitutive gene
    return jnp.array([F_b,
            F_switch,
            F_int,
            F_cat,
            F_prot])

# ode
def punisher_b_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # get the concentration of the integrase mRNA,scaled to account for translation by multiple ribosomes
    # same operon as the switch gene, but possibly not the same length in codons => rescaling m_switch
    m_int = x[name2pos['m_switch']]*par['n_int']/par['n_switch']

    # RETURN THE ODE
    return [# mRNAs
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] - (par['b_b'] + l) * x[name2pos['m_b']],
            par['func_switch'] * l * F[name2pos['F_switch']] * par['c_switch'] * par['a_switch'] - (par['b_switch'] + l) * x[name2pos['m_switch']],
            0, # integrase expressed from the same operon as the switch protein, not its own gene
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],  # NOTE: GENE COPY NO. GIVEN BY CAT_PB!
            par['func_prot'] * l * F[name2pos['F_prot']] * par['c_prot'] * par['a_prot'] - (par['b_prot'] + l) * x[name2pos['m_prot']],
            # proteins
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R - (l + par['d_b']*p_prot) * x[name2pos['p_b']],
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R - (l + par['d_switch']*p_prot) * x[name2pos['p_switch']],
            (e / par['n_int']) * (m_int / k_het[name2pos['k_int']] / D) * R - (l + par['d_int']*p_prot) * x[name2pos['p_int']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']*p_prot) * x[name2pos['p_cat']],
            (e / par['n_prot']) * (x[name2pos['m_prot']] / k_het[name2pos['k_prot']] / D) * R - (l + par['d_prot']*p_prot) * x[name2pos['p_prot']],
            # cat gene state with regard to the integrase
            # PB/PBI: fully functional (with attP and attB sites) free or with integrase bound
            -par['k_sxf']*int_cat_hill*x[name2pos['cat_pb']]+par['k_sxr']*x[name2pos['cat_lri1']],
            # LRI1: non-functional (with attL and attR sites) with integrase bound in active conformation - strands synapsed together
            (par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']] - par['k_sxr'] * x[name2pos['cat_lri1']]) + (
                -par['k_+syn'] * x[name2pos['cat_lri1']] - l*x[name2pos['cat_lri1']]) # no replenishment from LRI2/LR as strands diffuse away
    ]

# stochastic reaction propensities for hybrid tau-leaping simulations
def punisher_b_v(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            mRNA_count_scales, # scaling factors for mRNA counts
            par,  # system parameters
            name2pos
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # get the concentration of the integrase mRNA,scaled to account for translation by multiple ribosomes
    # same operon as the switch gene, but possibly not the same length in codons => rescaling m_switch
    m_int = x[name2pos['m_switch']] / mRNA_count_scales[name2pos['mscale_switch']] * mRNA_count_scales[name2pos['mscale_int']]

    # RETURN THE PROPENSITIES
    return [
            # synthesis, degradation, dilution of burdensome synthetic gene mRNA
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] / mRNA_count_scales[name2pos['mscale_b']],
            par['b_b'] * x[name2pos['m_b']] / mRNA_count_scales[name2pos['mscale_b']],
            l * x[name2pos['m_b']] / mRNA_count_scales[name2pos['mscale_b']],
            # synthesis, degradation, dilution of switch gene mRNA
            par['func_switch'] * l * F[name2pos['F_switch']] * par['c_switch'] * par['a_switch'] / mRNA_count_scales[name2pos['mscale_switch']],
            par['b_switch'] * x[name2pos['m_switch']] / mRNA_count_scales[name2pos['mscale_switch']],
            l * x[name2pos['m_switch']] / mRNA_count_scales[name2pos['mscale_switch']],
            # synthesis, degradation, dilution of integrase gene mRNA
            0,  # integrase expressed from the same operon as the switch protein, not its own gene
            0,  # integrase expressed from the same operon as the switch protein, not its own gene
            0,  # integrase expressed from the same operon as the switch protein, not its own gene
            # synthesis, degradation, dilution of cat gene mRNA
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] / mRNA_count_scales[name2pos['mscale_cat']],
            par['b_cat'] * x[name2pos['m_cat']] / mRNA_count_scales[name2pos['mscale_cat']],
            l * x[name2pos['m_cat']] / mRNA_count_scales[name2pos['mscale_cat']],
            # synthesis, degradation, dilution of protease gene mRNA
            par['func_prot'] * l * F[name2pos['F_prot']] * par['c_prot'] * par['a_prot'] / mRNA_count_scales[name2pos['mscale_prot']],
            par['b_prot'] * x[name2pos['m_prot']] / mRNA_count_scales[name2pos['mscale_prot']],
            l * x[name2pos['m_prot']] / mRNA_count_scales[name2pos['mscale_prot']],
            #
            # synthesis, degradation, dilution of burdensome synthetic gene protein
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R,
            par['d_b']*p_prot * x[name2pos['p_b']],
            l * x[name2pos['p_b']],
            # synthesis, degradation, dilution of switch gene protein
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R,
            par['d_switch']*p_prot * x[name2pos['p_switch']],
            l * x[name2pos['p_switch']],
            # synthesis, degradation, dilution of integrase gene protein
            (e / par['n_int']) * (m_int / k_het[name2pos['k_int']] / D) * R,
            par['d_int']*p_prot * x[name2pos['p_int']],
            l * x[name2pos['p_int']],
            # synthesis, degradation, dilution of cat gene protein
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R,
            par['d_cat']*p_prot * x[name2pos['p_cat']],
            l * x[name2pos['p_cat']],
            # synthesis, degradation, dilution of protease protein
            (e / par['n_prot']) * (x[name2pos['m_prot']] / k_het[name2pos['k_prot']] / D) * R,
            par['d_prot']*p_prot * x[name2pos['p_prot']],
            l * x[name2pos['p_prot']],
            #
            # cat gene state with regard to the integrase
            # CAT gene forward strain exchange: from cat_pb to cat_lri1
            par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']],
            # CAT gene reverse strain exchange: from cat_lri1 to cat_pb
            par['k_sxr'] * x[name2pos['cat_lri1']],
            # LR site-integrase dissociation due to conformation change
            par['k_+syn'] * x[name2pos['cat_lri1']],
            # LR site-integrase dissociation due to plasmid replicatrion
            l * x[name2pos['cat_lri1']]
        ]


# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def punisher_b_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                    e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, add the value for the integrase co-expressed with the switch gene
    k_int = (par['k-_int'] + e / par['n_int']) / par['k+_int']
    m_int = x[name2pos['m_switch']]*par['n_int']/par['n_switch']
    return m_het_div_k_het_uncorr + m_int / k_int


# PUNISHER - WITH SWITCH PROT. AND INTEGRASE EXPRESSED SEPARATELY - AND A SINGLE BURDENSOME GENE [tau-leap compatible]--
def punisher_sep_b_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['b',
             'switch', 'int', 'cat', 'prot']  # names of genes in the circuit
    miscs = ['cat_pb',
             'cat_lri1']  # names of miscellaneous species involved in the circuit. Here, cat gene states with respect to the integrase
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] = i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] = i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # punisher
    default_par['c_switch'] = 1  # gene concentration (nM)
    default_par['a_switch'] = 150  # promoter strength (unitless)
    default_par['c_int'] = 1  # gene concentration (nM)
    default_par['a_int'] = 150  # promoter strength (unitless)
    default_par['c_cat'] = 1  # gene concentration (nM)
    default_par['a_cat'] = 5000  # promoter strength (unitless)

    # transcription regulation function
    default_par['K_switch'] = 100
    default_par['eta_switch'] = 2
    default_par['baseline_switch'] = 0.1
    default_par['p_switch_ac_frac'] = 1  # ACTIVE fraction of protein (i.e. share of molecules bound by the inducer)

    # integrase action
    default_par['K_bI~'] = 100  # dissociation constant for the integrase and DNA (nM)
    default_par['k_sxf'] = 6  # forward strain exchange rate by the integrase(1/h)
    default_par['k_sxr'] = 2.14  # reverse strain exchange rate by the integrase(1/h)
    default_par['k_+syn'] = 0.006  # forward synaptic conformational change rate (1/h)

    # initial conditions for the state of cat gene with respect to the integrase
    default_init_conds['cat_pb'] = 10  # originally, the gene is in functional state (with attP and attB sites)
    default_init_conds['cat_lri1'] = 0  # originally, the gene is in functional state (with attP and attB sites)
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # give cat gene states same colours as cat mRNA and protein but different dashes
    circuit_styles['colours']['cat_pb'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['cat_lri1'] = circuit_styles['colours']['cat']

    circuit_styles['dashes']['cat_pb'] = 'solid'  # fully functional gene => solid dash
    circuit_styles['dashes']['cat_lri1'] = 'dashed'  # non-functional => non-solid
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def punisher_sep_b_F_calc(t ,x, par, name2pos):
    F_b = 1 # constitutive gene
    F_cat = 1 # constitutive gene

    # switch and integrase genes regulated by p_switch
    p_switch_dependent_term = (x[name2pos['p_switch']]*par['p_switch_ac_frac']/par['K_switch'])**par['eta_switch']
    F_switch = par['baseline_switch'] + (1 - par['baseline_switch']) * (p_switch_dependent_term/(p_switch_dependent_term+1))
    F_int = F_switch    # integrase co-regulated with the switch protein
    F_prot = 1 # constitutive gene
    return jnp.array([F_b,
            F_switch,
            F_int,
            F_cat,
            F_prot])

# ode
def punisher_sep_b_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # RETURN THE ODE
    return [# mRNAs
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] - (par['b_b'] + l) * x[name2pos['m_b']],
            par['func_switch'] * l * F[name2pos['F_switch']] * par['c_switch'] * par['a_switch'] - (par['b_switch'] + l) * x[name2pos['m_switch']],
            par['func_int'] * l * F[name2pos['F_int']] * par['c_int'] * par['a_int'] - (par['b_int'] + l) * x[name2pos['m_int']],
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],  # NOTE: GENE COPY NO. GIVEN BY CAT_PB!
            par['func_prot'] * l * F[name2pos['F_prot']] * par['c_prot'] * par['a_prot'] - (par['b_prot'] + l) * x[name2pos['m_prot']],
            # proteins
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R - (l + par['d_b']*p_prot) * x[name2pos['p_b']],
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R - (l + par['d_switch']*p_prot) * x[name2pos['p_switch']],
            (e / par['n_int']) * (x[name2pos['m_int']] / k_het[name2pos['k_int']] / D) * R - (l + par['d_int']*p_prot) * x[name2pos['p_int']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']*p_prot) * x[name2pos['p_cat']],
            (e / par['n_prot']) * (x[name2pos['m_prot']] / k_het[name2pos['k_prot']] / D) * R - (l + par['d_prot']*p_prot) * x[name2pos['p_prot']],
            # cat gene state with regard to the integrase
            # PB/PBI: fully functional (with attP and attB sites) free or with integrase bound
            -par['k_sxf']*int_cat_hill*x[name2pos['cat_pb']]+par['k_sxr']*x[name2pos['cat_lri1']],
            # LRI1: non-functional (with attL and attR sites) with integrase bound in active conformation - strands synapsed together
            (par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']] - par['k_sxr'] * x[name2pos['cat_lri1']]) + (
                -par['k_+syn'] * x[name2pos['cat_lri1']] - l*x[name2pos['cat_lri1']]) # no replenishment from LRI2/LR as strands diffuse away
    ]

# stochastic reaction propensities for hybrid tau-leaping simulations
def punisher_sep_b_v(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            mRNA_count_scales, # scaling factors for mRNA counts
            par,  # system parameters
            name2pos
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # RETURN THE PROPENSITIES
    return [
            # synthesis, degradation, dilution of burdensome synthetic gene mRNA
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] / mRNA_count_scales[name2pos['mscale_b']],
            par['b_b'] * x[name2pos['m_b']] / mRNA_count_scales[name2pos['mscale_b']],
            l * x[name2pos['m_b']] / mRNA_count_scales[name2pos['mscale_b']],
            # synthesis, degradation, dilution of switch gene mRNA
            par['func_switch'] * l * F[name2pos['F_switch']] * par['c_switch'] * par['a_switch'] / mRNA_count_scales[name2pos['mscale_switch']],
            par['b_switch'] * x[name2pos['m_switch']] / mRNA_count_scales[name2pos['mscale_switch']],
            l * x[name2pos['m_switch']] / mRNA_count_scales[name2pos['mscale_switch']],
            # synthesis, degradation, dilution of integrase gene mRNA
            par['func_int'] * l * F[name2pos['F_int']] * par['c_int'] * par['a_int'] / mRNA_count_scales[name2pos['mscale_int']],
            par['b_int'] * x[name2pos['m_int']] / mRNA_count_scales[name2pos['mscale_int']],
            l * x[name2pos['m_int']] / mRNA_count_scales[name2pos['mscale_int']],
            # synthesis, degradation, dilution of cat gene mRNA
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] / mRNA_count_scales[name2pos['mscale_cat']],
            par['b_cat'] * x[name2pos['m_cat']] / mRNA_count_scales[name2pos['mscale_cat']],
            l * x[name2pos['m_cat']] / mRNA_count_scales[name2pos['mscale_cat']],
            # synthesis, degradation, dilution of protease gene mRNA
            par['func_prot'] * l * F[name2pos['F_prot']] * par['c_prot'] * par['a_prot'] / mRNA_count_scales[name2pos['mscale_prot']],
            par['b_prot'] * x[name2pos['m_prot']] / mRNA_count_scales[name2pos['mscale_prot']],
            l * x[name2pos['m_prot']] / mRNA_count_scales[name2pos['mscale_prot']],
            #
            # synthesis, degradation, dilution of burdensome synthetic gene protein
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R,
            par['d_b']*p_prot * x[name2pos['p_b']],
            l * x[name2pos['p_b']],
            # synthesis, degradation, dilution of switch gene protein
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R,
            par['d_switch']*p_prot * x[name2pos['p_switch']],
            l * x[name2pos['p_switch']],
            # synthesis, degradation, dilution of integrase gene protein
            (e / par['n_int']) * (x[name2pos['m_int']] / k_het[name2pos['k_int']] / D) * R,
            par['d_int']*p_prot * x[name2pos['p_int']],
            l * x[name2pos['p_int']],
            # synthesis, degradation, dilution of cat gene protein
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R,
            par['d_cat']*p_prot * x[name2pos['p_cat']],
            l * x[name2pos['p_cat']],
            # synthesis, degradation, dilution of protease protein
            (e / par['n_prot']) * (x[name2pos['m_prot']] / k_het[name2pos['k_prot']] / D) * R,
            par['d_prot']*p_prot * x[name2pos['p_prot']],
            l * x[name2pos['p_prot']],
            #
            # cat gene state with regard to the integrase
            # CAT gene forward strain exchange: from cat_pb to cat_lri1
            par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']],
            # CAT gene reverse strain exchange: from cat_lri1 to cat_pb
            par['k_sxr'] * x[name2pos['cat_lri1']],
            # LR site-integrase dissociation due to conformation change
            par['k_+syn'] * x[name2pos['cat_lri1']],
            # LR site-integrase dissociation due to plasmid replicatrion
            l * x[name2pos['cat_lri1']]
        ]


# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def punisher_sep_b_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                        e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, no correction as no co-expression
    return m_het_div_k_het_uncorr


# PUNISHER (WITH AN EXPLICIT COPY NUMBER CONTROL SYSTEM) AND A SINGLE BURDENSOME GENE ----------------------------------
# TAU-LEAPING SIMULATIONS IMPLEMENTED FOR THE VARIABLE CELL VOLUME CASE
def punisher_cnc_b_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    # names of genes in the circuit
    genes = ['b',
             'switch', 'int', 'cat', 'prot']
    # names of miscellaneous species involved in the circuit. Here, the plasmid hosting the Punisher...
    # 1) with attP and attB sites around the CAT gene
    # 2) with CAT gene in the integhrase-attL-attR site complex
    # 3) CAT gene excised
    # plus the INHIBITOR  of plasmid replication
    miscs = ['cat_pb', 'cat_lri1', 'no_cat', 'inh']
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
        name2pos['rep_vol_'+genes[i]] = -1 # by default, the gene is not chromosomal => position -1 assigned
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)
        default_par['mean_rep_phase_' + gene] = -1.0  # mean replication phase of the gene's plasmid - negative by default, meaning gene not chromosomal
        default_par['stdev_rep_phase_' + gene] = 0.23  # standard deviation of the replication phase of the gene's plasmid - 0.23 from Walker et al., BMC Biology, 2016

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # punisher
    default_par['c_switch'] = 1  # gene concentration (nM)
    default_par['a_switch'] = 150  # promoter strength (unitless)
    default_par['c_int'] = 0  # integrase expressed from the same operon as the switch protein
    default_par['a_int'] = 0 # integrase expressed from the same operon as the switch protein
    default_par['c_cat'] = 1  # gene concentration (nM)
    default_par['a_cat'] = 5000  # promoter strength (unitless)

    # transcription regulation function
    default_par['K_switch'] = 100
    default_par['eta_switch'] = 2
    default_par['baseline_switch'] = 0.1
    default_par['p_switch_ac_frac'] = 1  # ACTIVE fraction of protein (i.e. share of molecules bound by the inducer)

    # integrase action
    default_par['K_bI~'] = 100 # dissociation constant for the integrase and DNA (nM)
    default_par['k_sxf'] = 6 # forward strain exchange rate by the integrase(1/h)
    default_par['k_sxr'] = 2.14 # reverse strain exchange rate by the integrase(1/h)
    default_par['k_+syn'] = 0.006 # forward synaptic conformational change rate (1/h)

    # initial conditions for the state of cat gene with respect to the integrase
    default_init_conds['cat_pb'] = 10 # originally, the gene is in functional state (with attP and attB sites)
    default_init_conds['cat_lri1'] = 0 # originally, the gene is in functional state (with attP and attB sites)
    default_init_conds['no_cat'] = 0 # originally, the gene is in functional state (with attP and attB sites)

    # plasmid replication control
    default_par['k_tr'] = 130.2  # plasmid replication rate (1/h)
    default_par['a_inh'] = 948  # inhibitor synthesis rate per plasmid copy (1/h)
    default_par['b_inh'] = 74.976  # inhibitor degradation rate (1/h)
    default_par['n_inh'] = 10  # number of steps of replication initiation at which inhibition can happen
    default_par['K_inh'] = 214.05  # replication inhibition constant (nM)

    # replication of the chromosomally integrated burdensome gene
    default_par['mean_rep_phase_b'] = 0.5  # mean replication phase of the gene's plasmid
    default_par['stdev_rep_phase_b'] = 0.23  # standard deviation of the replication phase of the gene's plasmid
    name2pos['rep_vol_b']=0  # position of the volume of the plasmid carrying the burdensome gene in the replication volumes vector

    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles={'colours':{}, 'dashes':{}} # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # give cat gene states same colours as cat mRNA and protein but different dashes
    circuit_styles['colours']['cat_pb'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['cat_lri1'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['no_cat'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['inh'] = '#0072BD' # different colour for the inhibitor

    circuit_styles['dashes']['cat_pb'] = 'solid' # fully functional gene => solid
    circuit_styles['dashes']['cat_lri1'] = 'dashed' # non-functional => dashed
    circuit_styles['dashes']['no_cat'] = 'dotted'  # non-functional and irreversibly excised => dotted
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def punisher_cnc_b_F_calc(t ,x, par, name2pos):
    F_b = 1 # constitutive gene
    F_cat = 1 # constitutive gene

    # switch and integrase genes regulated by p_switch
    p_switch_dependent_term = (x[name2pos['p_switch']]*par['p_switch_ac_frac']/par['K_switch'])**par['eta_switch']
    F_switch = par['baseline_switch'] + (1 - par['baseline_switch']) * (p_switch_dependent_term/(p_switch_dependent_term+1))
    F_int = 0    # integrase expressed from the same operon as the switch protein, not its own gene
    F_prot = 1 # constitutive gene
    return jnp.array([F_b,
            F_switch,
            F_int,
            F_cat,
            F_prot])

# ode
def punisher_cnc_b_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # get the concentration of the integrase mRNA,scaled to account for translation by multiple ribosomes
    # same operon as the switch gene, but possibly not the same length in codons => rescaling m_switch
    m_int = x[name2pos['m_switch']]*par['n_int']/par['n_switch']

    # get the total concentration of the plasmid with Punisher genes
    c_total = x[name2pos['cat_pb']] + x[name2pos['cat_lri1']] + x[name2pos['no_cat']]
    # get the plasmid replication rate
    rep_rate = par['k_tr'] * (par['K_inh']/(x[name2pos['inh']]+par['K_inh']))**par['n_inh']   # hyperbolic inhibition

    # RETURN THE ODE
    return [# mRNAs
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] - (par['b_b'] + l) * x[name2pos['m_b']],  # NOT c_total: burdensome gene considered separately
            par['func_switch'] * l * F[name2pos['F_switch']] * c_total * par['a_switch'] - (par['b_switch'] + l) * x[name2pos['m_switch']], # mind the c_total!
            0, # integrase expressed from the same operon as the switch protein, not its own gene
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],  # NOTE: GENE COPY NO. GIVEN BY CAT_PB! (not c_total)
            par['func_prot'] * l * F[name2pos['F_prot']] * c_total * par['a_prot'] - (par['b_prot'] + l) * x[name2pos['m_prot']],   # mind the c_total!
            # proteins
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R - (l + par['d_b']*p_prot) * x[name2pos['p_b']],
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R - (l + par['d_switch']*p_prot) * x[name2pos['p_switch']],
            (e / par['n_int']) * (m_int / k_het[name2pos['k_int']] / D) * R - (l + par['d_int']*p_prot) * x[name2pos['p_int']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']*p_prot) * x[name2pos['p_cat']],
            (e / par['n_prot']) * (x[name2pos['m_prot']] / k_het[name2pos['k_prot']] / D) * R - (l + par['d_prot']*p_prot) * x[name2pos['p_prot']],
            # cat gene state with regard to the integrase
            # PB/PBI: fully functional (with attP and attB sites) free or with integrase bound
            rep_rate*x[name2pos['cat_pb']] - l*x[name2pos['cat_pb']] \
                - par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']] + par['k_sxr'] * x[name2pos['cat_lri1']],
            # LRI1: non-functional (with attL and attR sites) with integrase bound in active conformation - strands synapsed together
            # plasmid replication machinery breaks up the complex => - sign at replication term
            -rep_rate*x[name2pos['cat_lri1']] - l*x[name2pos['cat_lri1']] \
                + (par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']] - par['k_sxr'] * x[name2pos['cat_lri1']]) \
                - par['k_+syn']* x[name2pos['cat_lri1']],
            # plasmid copies without CAT gene DNA
            #  plasmid replication machinery breaks up the complex => also replenished as cat_lri1 is replicated
            rep_rate*x[name2pos['no_cat']] - l*x[name2pos['no_cat']] \
                + par['k_+syn']* x[name2pos['cat_lri1']] + 2*rep_rate*x[name2pos['cat_lri1']],
            # inhibitor of plasmid replication
            par['a_inh'] * c_total - (l + par['b_inh']) * x[name2pos['inh']]
    ]

# stochastic reaction propensities for hybrid tau-leaping simulations
def punisher_cnc_b_v_varvol(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            mRNA_count_scales, # scaling factors for mRNA counts
            par,  # system parameters
            name2pos, # name to position decoder
            V, # cell volume
            rep_vols, # volumes at which chromosomally integrated synthetic genes are replicated
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # get the concentration of the integrase mRNA,scaled to account for translation by multiple ribosomes
    # same operon as the switch gene, but possibly not the same length in codons => rescaling m_switch
    m_int = x[name2pos['m_switch']] / mRNA_count_scales[name2pos['mscale_switch']] * mRNA_count_scales[name2pos['mscale_int']]

    # get the total concentration of the plasmid with Punisher genes
    c_total = x[name2pos['cat_pb']] + x[name2pos['cat_lri1']] + x[name2pos['no_cat']]
    # get the plasmid replication rate
    rep_rate = par['k_tr'] * (par['K_inh'] / (x[name2pos['inh']] + par['K_inh'])) ** par['n_inh']  # hyperbolic inhibition

    # burdensome gene chromosomally integrated => handle its concentration separately
    c_b=(1+(V>=rep_vols[name2pos['rep_vol_b']]))/V # one copy per cell, or two when replicated

    # RETURN THE PROPENSITIES
    return [
            # synthesis and degradation of burdensome synthetic gene mRNA
            par['func_b'] * l * F[name2pos['F_b']] * c_b * par['a_b'] / mRNA_count_scales[name2pos['mscale_b']],
            par['b_b'] * x[name2pos['m_b']] / mRNA_count_scales[name2pos['mscale_b']],
            # synthesis and degradation of switch gene mRNA
            par['func_switch'] * l * F[name2pos['F_switch']] * c_total * par['a_switch'] / mRNA_count_scales[name2pos['mscale_switch']],
            par['b_switch'] * x[name2pos['m_switch']] / mRNA_count_scales[name2pos['mscale_switch']],
            # synthesis and degradation of integrase gene mRNA
            0,  # integrase expressed from the same operon as the switch protein, not its own gene
            0,  # integrase expressed from the same operon as the switch protein, not its own gene
            # synthesis and degradation of cat gene mRNA
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] / mRNA_count_scales[name2pos['mscale_cat']],
            par['b_cat'] * x[name2pos['m_cat']] / mRNA_count_scales[name2pos['mscale_cat']],
            # synthesis and degradation of protease gene mRNA
            par['func_prot'] * l * F[name2pos['F_prot']] * c_total * par['a_prot'] / mRNA_count_scales[name2pos['mscale_prot']],
            par['b_prot'] * x[name2pos['m_prot']] / mRNA_count_scales[name2pos['mscale_prot']],
            #
            # synthesis and degradation of burdensome synthetic gene protein
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R,
            par['d_b']*p_prot * x[name2pos['p_b']],
            # synthesis and degradation of switch gene protein
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R,
            par['d_switch']*p_prot * x[name2pos['p_switch']],
            # synthesis and degradation of integrase gene protein
            (e / par['n_int']) * (m_int / k_het[name2pos['k_int']] / D) * R,
            par['d_int']*p_prot * x[name2pos['p_int']],
            # synthesis and degradation of cat gene protein
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R,
            par['d_cat']*p_prot * x[name2pos['p_cat']],
            # synthesis and degradation of protease protein
            (e / par['n_prot']) * (x[name2pos['m_prot']] / k_het[name2pos['k_prot']] / D) * R,
            par['d_prot']*p_prot * x[name2pos['p_prot']],
            #
            # integrase action
            # CAT gene forward strain exchange: from cat_pb to cat_lri1
            par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']],
            # CAT gene reverse strain exchange: from cat_lri1 to cat_pb
            par['k_sxr'] * x[name2pos['cat_lri1']],
            # LR site-integrase dissociation due to conformation change
            par['k_+syn'] * x[name2pos['cat_lri1']],
            #
            # plasmid replication
            # replication of plasmid with LR site-integrase complex - causes formation of no-CAT plasmids
            rep_rate * x[name2pos['cat_lri1']],
            #  replication of plasmid with CAT gene
            rep_rate * x[name2pos['cat_pb']],
            # replication of plasmid without CAT gene
            rep_rate * x[name2pos['no_cat']],
            #
            # plasmid replication inhibitor dynamics
            # synthesis of the inhibitor
            par['a_inh'] * c_total,
            # degradation of the inhibitor
            par['b_inh'] * x[name2pos['inh']]
        ]


# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def punisher_cnc_b_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                    e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, add the value for the integrase co-expressed with the switch gene
    k_int = (par['k-_int'] + e / par['n_int']) / par['k+_int']
    m_int = x[name2pos['m_switch']]*par['n_int']/par['n_switch']
    return m_het_div_k_het_uncorr + m_int / k_int


# PUNISHER WITH TWO TOGGLE SWITCHES ------------------------------------------------------------------------------------
def twotoggles_punisher_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['tog11','tog12','tog21','tog22',  # toggle switches
             'switch', 'int', 'cat', 'prot']  # punisher
    miscs = ['cat_pb', 'cat_lri1']  # names of miscellaneous species involved in the circuit. Here, cat gene states with respect to the integrase
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] = i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # toggle switch parameters
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            default_par['c_tog' + str(togswitchnum) + str(toggenenum)] = 1  # copy no. (nM)
            default_par['a_tog' + str(togswitchnum) + str(toggenenum)] = 10 ** 5 / 2  # promoter strength (unitless)

            # transcription regulation function
            reg_func_string = 'dna(tog' + str(togswitchnum) + str(toggenenum) + '):p_tog' + str(togswitchnum) + str(
                toggenenum % 2 + 1)
            default_par['K_' + reg_func_string] = 2000  # half-saturation constant
            default_par['eta_' + reg_func_string] = 2  # Hill coefficient
            default_par['baseline_tog' + str(togswitchnum) + str(
                toggenenum)] = 0.05  # baseline transcription activation due to leakiness
            default_par['p_tog' + str(togswitchnum) + str(
                toggenenum) + '_ac_frac'] = 1  # active fraction of protein (i.e. share of molecules NOT bound by the inducer)

        # break symmetry for each of the toggle switches
        default_init_conds['m_tog' + str(togswitchnum) + '1'] = 500

    # punisher
    default_par['c_switch'] = 1  # gene concentration (nM)
    default_par['a_switch'] = 150  # promoter strength (unitless)
    default_par['c_int'] = 1  # gene concentration (nM)
    default_par['a_int'] = 1000  # promoter strength (unitless)
    default_par['d_int'] = 6  # integrase protein degradation rate (to avoid unnecessary punishment)
    default_par['c_cat'] = 1  # gene concentration (nM)
    default_par['a_cat'] = 5000  # promoter strength (unitless)

    # transcription regulation function
    default_par['K_switch'] = 100
    default_par['eta_switch'] = 2
    default_par['baseline_switch'] = 0.1
    default_par['p_switch_ac_frac'] = 1  # ACTIVE fraction of protein (i.e. share of molecules bound by the inducer)

    # integrase action
    default_par['K_bI~'] = 100  # dissociation constant for the integrase and DNA (nM)
    default_par['k_sxf'] = 6  # forward strain exchange rate by the integrase(1/h)
    default_par['k_sxr'] = 2.14  # reverse strain exchange rate by the integrase(1/h)
    default_par['k_+syn'] = 0.006  # forward synaptic conformational change rate (1/h)

    # initial conditions for the state of cat gene with respect to the integrase
    default_init_conds['cat_pb'] = 10  # originally, the gene is in functional state (with attP and attB sites)
    default_init_conds['cat_lri1'] = 0  # originally, the gene is in functional state (with attP and attB sites)
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["dodgerblue", "chocolate", "orchid", "crimson",
                       "darkcyan", "darkgoldenrod", "darkviolet", "firebrick"]  # match default palette to genes looping over the five colours we defined
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # give cat gene states same colours as cat mRNA and protein but different dashes
    circuit_styles['colours']['cat_pb'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['cat_lri1'] = circuit_styles['colours']['cat']

    circuit_styles['dashes']['cat_pb'] = 'solid'  # fully functional gene => solid dash
    circuit_styles['dashes']['cat_lri1'] = 'dashed'  # non-functional => non-solid
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles


# transcription regulation functions
def twotoggles_punisher_F_calc(t ,x, par, name2pos):
    F_cat = 1 # cat is a constitutive gene

    # toggle switch 1
    F_tog11 = par['baseline_tog11'] + (1 - par['baseline_tog11']) * \
        1 / (1 +
             (x[name2pos['p_tog12']] * par['p_tog12_ac_frac'] / par['K_dna(tog11):p_tog12']) ** par['eta_dna(tog11):p_tog12'])  # tog11: regulated by p_tog12

    F_tog12 = par['baseline_tog12'] + (1 - par['baseline_tog12']) * \
                1 / (1 + (x[name2pos['p_tog11']] * par['p_tog11_ac_frac'] / par[
                'K_dna(tog12):p_tog11']) ** par['eta_dna(tog12):p_tog11'])  # tog12: regulated by p_tog11

    # toggle switch 2
    F_tog21 = par['baseline_tog21'] + (1 - par['baseline_tog21']) * \
                1 / (1 + (x[name2pos['p_tog22']] * par['p_tog22_ac_frac'] / par[
                'K_dna(tog21):p_tog22']) ** par['eta_dna(tog21):p_tog22'])  # tog21: regulated by p_tog22
    F_tog22 = par['baseline_tog22'] + (1 - par['baseline_tog22']) * \
                1 / (1 + (x[name2pos['p_tog21']] * par['p_tog21_ac_frac'] / par[
                'K_dna(tog22):p_tog21']) ** par['eta_dna(tog22):p_tog21'])  # tog22: regulated by p_tog21

    # switch and integrase genes regulated by p_switch
    p_switch_dependent_term = (x[name2pos['p_switch']]*par['p_switch_ac_frac']/par['K_switch'])**par['eta_switch']
    F_switch = par['baseline_switch'] + (1 - par['baseline_switch']) * (p_switch_dependent_term/(p_switch_dependent_term+1))
    F_int = 0    # integrase expressed from the same operon as the switch protein, not its own gene
    F_prot = 1 # constitutive gene
    return jnp.array([F_tog11,
                      F_tog12,
                      F_tog21,
                      F_tog22,
                      F_switch,
                      F_int,
                      F_cat,
                      F_prot])


# ode
def twotoggles_punisher_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # get the concentration of the integrase mRNA,scaled to account for translation by multiple ribosomes
    # same operon as the switch gene, but possibly not the same length in codons => rescaling m_switch
    m_int = x[name2pos['m_switch']] * par['n_int'] / par['n_switch']

    # RETURN THE ODE
    return [# mRNAs
            par['func_tog11'] * l * F[name2pos['F_tog11']] * par['c_tog11'] * par['a_tog11'] - (par['b_tog11'] + l) * x[name2pos['m_tog11']],
            par['func_tog12'] * l * F[name2pos['F_tog12']] * par['c_tog12'] * par['a_tog12'] - (par['b_tog12'] + l) * x[name2pos['m_tog12']],
            par['func_tog21'] * l * F[name2pos['F_tog21']] * par['c_tog21'] * par['a_tog21'] - (par['b_tog21'] + l) * x[name2pos['m_tog21']],
            par['func_tog22'] * l * F[name2pos['F_tog22']] * par['c_tog22'] * par['a_tog22'] - (par['b_tog22'] + l) * x[name2pos['m_tog22']],
            par['func_switch'] * l * F[name2pos['F_switch']] * par['c_switch'] * par['a_switch'] - (par['b_switch'] + l) * x[name2pos['m_switch']],
            0,  # integrase expressed from the same operon as the switch protein, not its own gene
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],  # NOTE: GENE COPY NO. GIVEN BY CAT_PB!
            par['func_prot'] * l * F[name2pos['F_prot']] * par['c_prot'] * par['a_prot'] - (par['b_prot'] + l) * x[name2pos['m_prot']],
            # proteins
            (e / par['n_tog11']) * (x[name2pos['m_tog11']] / k_het[name2pos['k_tog11']] / D) * R - (l + par['d_tog11']*p_prot) * x[name2pos['p_tog11']],
            (e / par['n_tog12']) * (x[name2pos['m_tog12']] / k_het[name2pos['k_tog12']] / D) * R - (l + par['d_tog12']*p_prot) * x[name2pos['p_tog12']],
            (e / par['n_tog21']) * (x[name2pos['m_tog21']] / k_het[name2pos['k_tog21']] / D) * R - (l + par['d_tog21']*p_prot) * x[name2pos['p_tog21']],
            (e / par['n_tog22']) * (x[name2pos['m_tog22']] / k_het[name2pos['k_tog22']] / D) * R - (l + par['d_tog22']*p_prot) * x[name2pos['p_tog22']],
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R - (l + par['d_switch']*p_prot) * x[name2pos['p_switch']],
            (e / par['n_int']) * (m_int / k_het[name2pos['k_int']] / D) * R - (l + par['d_int']*p_prot) * x[name2pos['p_int']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']*p_prot) * x[name2pos['p_cat']],
            (e / par['n_prot']) * (x[name2pos['m_prot']] / k_het[name2pos['k_prot']] / D) * R - (l + par['d_prot']*p_prot) * x[name2pos['p_prot']],
            # cat gene state with regard to the integrase
            # PB/PBI: fully functional (with attP and attB sites) free or with integrase bound
            -par['k_sxf']*int_cat_hill*x[name2pos['cat_pb']]+par['k_sxr']*x[name2pos['cat_lri1']],
            # LRI1: non-functional (with attL and attR sites) with integrase bound in active conformation - strands synapsed together
            (par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']] - par['k_sxr'] * x[name2pos['cat_lri1']]) + (
                -par['k_+syn'] * x[name2pos['cat_lri1']] - l*x[name2pos['cat_lri1']]) # no replenishment from LRI2/LR as strands diffuse away
    ]

# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def twotoggles_punisher_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                    e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, add the value for the integrase co-expressed with the switch gene
    k_int = (par['k-_int'] + e / par['n_int']) / par['k+_int']
    m_int = x[name2pos['m_switch']]*par['n_int']/par['n_switch']
    return m_het_div_k_het_uncorr + m_int / k_int


# TWO TOGGLE SWITCHES ONLY----------------------------------------------------------------------------------------------
def twotoggles_only_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['tog11','tog12','tog21','tog22']  # toggle switches
    miscs = []  # names of miscellaneous species involved in the circuit
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] = i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # toggle switch parameters
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            default_par['c_tog' + str(togswitchnum) + str(toggenenum)] = 1  # copy no. (nM)
            default_par['a_tog' + str(togswitchnum) + str(toggenenum)] = 10 ** 5 / 2  # promoter strength (unitless)

            # transcription regulation function
            reg_func_string = 'dna(tog' + str(togswitchnum) + str(toggenenum) + '):p_tog' + str(togswitchnum) + str(
                toggenenum % 2 + 1)
            default_par['K_' + reg_func_string] = 2000  # half-saturation constant
            default_par['eta_' + reg_func_string] = 2  # Hill coefficient
            default_par['baseline_tog' + str(togswitchnum) + str(
                toggenenum)] = 0.05  # baseline transcription activation due to leakiness
            default_par['p_tog' + str(togswitchnum) + str(
                toggenenum) + '_ac_frac'] = 1  # active fraction of protein (i.e. share of molecules NOT bound by the inducer)

        # break symmetry for each of the toggle switches
        default_init_conds['m_tog' + str(togswitchnum) + '1'] = 500
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["dodgerblue", "chocolate", "orchid", "crimson",
                       "darkcyan", "darkgoldenrod", "darkviolet", "firebrick"]  # match default palette to genes looping over the five colours we defined
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles


# transcription regulation functions
def twotoggles_only_F_calc(t ,x, par, name2pos):
    F_cat = 1 # cat is a constitutive gene

    # toggle switch 1
    F_tog11 = par['baseline_tog11'] + (1 - par['baseline_tog11']) * \
        1 / (1 +
             (x[name2pos['p_tog12']] * par['p_tog12_ac_frac'] / par['K_dna(tog11):p_tog12']) ** par['eta_dna(tog11):p_tog12'])  # tog11: regulated by p_tog12

    F_tog12 = par['baseline_tog12'] + (1 - par['baseline_tog12']) * \
                1 / (1 + (x[name2pos['p_tog11']] * par['p_tog11_ac_frac'] / par[
                'K_dna(tog12):p_tog11']) ** par['eta_dna(tog12):p_tog11'])  # tog12: regulated by p_tog11

    # toggle switch 2
    F_tog21 = par['baseline_tog21'] + (1 - par['baseline_tog21']) * \
                1 / (1 + (x[name2pos['p_tog22']] * par['p_tog22_ac_frac'] / par[
                'K_dna(tog21):p_tog22']) ** par['eta_dna(tog21):p_tog22'])  # tog21: regulated by p_tog22
    F_tog22 = par['baseline_tog22'] + (1 - par['baseline_tog22']) * \
                1 / (1 + (x[name2pos['p_tog21']] * par['p_tog21_ac_frac'] / par[
                'K_dna(tog22):p_tog21']) ** par['eta_dna(tog22):p_tog21'])  # tog22: regulated by p_tog21

    return jnp.array([F_tog11,
                      F_tog12,
                      F_tog21,
                      F_tog22])


# ode
def twotoggles_only_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_tog11'] * l * F[name2pos['F_tog11']] * par['c_tog11'] * par['a_tog11'] - (par['b_tog11'] + l) * x[name2pos['m_tog11']],
            par['func_tog12'] * l * F[name2pos['F_tog12']] * par['c_tog12'] * par['a_tog12'] - (par['b_tog12'] + l) * x[name2pos['m_tog12']],
            par['func_tog21'] * l * F[name2pos['F_tog21']] * par['c_tog21'] * par['a_tog21'] - (par['b_tog21'] + l) * x[name2pos['m_tog21']],
            par['func_tog22'] * l * F[name2pos['F_tog22']] * par['c_tog22'] * par['a_tog22'] - (par['b_tog22'] + l) * x[name2pos['m_tog22']],
            # proteins
            (e / par['n_tog11']) * (x[name2pos['m_tog11']] / k_het[name2pos['k_tog11']] / D) * R - (l + par['d_tog11']*p_prot) * x[name2pos['p_tog11']],
            (e / par['n_tog12']) * (x[name2pos['m_tog12']] / k_het[name2pos['k_tog12']] / D) * R - (l + par['d_tog12']*p_prot) * x[name2pos['p_tog12']],
            (e / par['n_tog21']) * (x[name2pos['m_tog21']] / k_het[name2pos['k_tog21']] / D) * R - (l + par['d_tog21']*p_prot) * x[name2pos['p_tog21']],
            (e / par['n_tog22']) * (x[name2pos['m_tog22']] / k_het[name2pos['k_tog22']] / D) * R - (l + par['d_tog22']*p_prot) * x[name2pos['p_tog22']],
    ]

# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def twotoggles_only_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                    e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, no correction needed
    return m_het_div_k_het_uncorr


# TWO TOGGLE SWITCHES WITH SYNTHETIC ADDICTION -------------------------------------------------------------------------
def twotoggles_add_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['tog11', 'tog12', 'tog21', 'tog22',  # toggle switches
             'cat']  # CAT (co-expressed from the same operons as the toggle switch genes)
    miscs = []  # names of miscellaneous species involved in the circuit. Here, none
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] = i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)
        default_par['g_' + gene] = 0.0  # synthetic protein's inteference with the cell's metabolic flux - zero by default (unitless)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # toggle switch parameters
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            default_par['c_tog' + str(togswitchnum) + str(toggenenum)] = 1  # copy no. (nM)
            default_par['a_tog' + str(togswitchnum) + str(toggenenum)] = 10 ** 5 / 2  # promoter strength (unitless)

            # transcription regulation function
            reg_func_string = 'dna(tog' + str(togswitchnum) + str(toggenenum) + '):p_tog' + str(togswitchnum) + str(
                toggenenum % 2 + 1)
            default_par['K_' + reg_func_string] = 2000  # half-saturation constant
            default_par['eta_' + reg_func_string] = 2  # Hill coefficient
            default_par['baseline_tog' + str(togswitchnum) + str(
                toggenenum)] = 0.05  # baseline transcription activation due to leakiness
            default_par['p_tog' + str(togswitchnum) + str(
                toggenenum) + '_ac_frac'] = 1  # active fraction of protein (i.e. share of molecules NOT bound by the inducer)

        # break symmetry for each of the toggle switches
        default_init_conds['m_tog' + str(togswitchnum) + '1'] = 500

    # co-expressed CAT gene
    default_par['n_cat'] = 300  # protein length (aa)
    default_par['d_cat'] = 0  # rate of active protein degradation - zero by default (/h)
    default_par['k+_cat11'] = 60  # ribosome binding rate (/h/nM)
    default_par['k-_cat11'] = 60  # ribosome unbinding rate (/h)
    default_par['k+_cat12'] = 60  # ribosome binding rate (/h/nM)
    default_par['k-_cat12'] = 60  # ribosome unbinding rate (/h)
    default_par['k+_cat21'] = 60  # ribosome binding rate (/h/nM)
    default_par['k-_cat21'] = 60  # ribosome unbinding rate (/h)
    default_par['k+_cat22'] = 60  # ribosome binding rate (/h/nM)
    default_par['k-_cat22'] = 60  # ribosome unbinding rate (/h)
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["dodgerblue", "chocolate", "orchid", "crimson",
                       "darkcyan", "darkgoldenrod", "darkviolet", "firebrick"]  # match default palette to genes looping over the five colours we defined
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # give cat gene states same colours as cat mRNA and protein but different dashes
    circuit_styles['colours']['cat_pb'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['cat_lri1'] = circuit_styles['colours']['cat']

    circuit_styles['dashes']['cat_pb'] = 'solid'  # fully functional gene => solid dash
    circuit_styles['dashes']['cat_lri1'] = 'dashed'  # non-functional => non-solid
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles


# transcription regulation functions
def twotoggles_add_F_calc(t ,x, par, name2pos):
    F_cat = 0 # irrelevant, as CAT co-expressed with toggle switch genes

    # toggle switch 1
    F_tog11 = par['baseline_tog11'] + (1 - par['baseline_tog11']) * \
        1 / (1 +
             (x[name2pos['p_tog12']] * par['p_tog12_ac_frac'] / par['K_dna(tog11):p_tog12']) ** par['eta_dna(tog11):p_tog12'])  # tog11: regulated by p_tog12

    F_tog12 = par['baseline_tog12'] + (1 - par['baseline_tog12']) * \
                1 / (1 + (x[name2pos['p_tog11']] * par['p_tog11_ac_frac'] / par[
                'K_dna(tog12):p_tog11']) ** par['eta_dna(tog12):p_tog11'])  # tog12: regulated by p_tog11

    # toggle switch 2
    F_tog21 = par['baseline_tog21'] + (1 - par['baseline_tog21']) * \
                1 / (1 + (x[name2pos['p_tog22']] * par['p_tog22_ac_frac'] / par[
                'K_dna(tog21):p_tog22']) ** par['eta_dna(tog21):p_tog22'])  # tog21: regulated by p_tog22
    F_tog22 = par['baseline_tog22'] + (1 - par['baseline_tog22']) * \
                1 / (1 + (x[name2pos['p_tog21']] * par['p_tog21_ac_frac'] / par[
                'K_dna(tog22):p_tog21']) ** par['eta_dna(tog22):p_tog21'])  # tog22: regulated by p_tog21

    return jnp.array([F_tog11,
                      F_tog12,
                      F_tog21,
                      F_tog22,
                      F_cat])


# ode
def twotoggles_add_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the concentration of the CAT mRNA,scaled to account for translation by multiple ribosomes
    # same operon as the four toggle genes, but possibly not the same length in codons => rescaling m_switch
    m_cat11 = x[name2pos['m_tog11']] * par['n_cat'] / par['n_tog11']
    m_cat12 = x[name2pos['m_tog12']] * par['n_cat'] / par['n_tog12']
    m_cat21 = x[name2pos['m_tog21']] * par['n_cat'] / par['n_tog21']
    m_cat22 = x[name2pos['m_tog22']] * par['n_cat'] / par['n_tog22']
    # get the RNA-ribosome dissociation constants for the CAT gene copies
    k_cat11 = (par['k-_cat11'] + e / par['n_cat']) / par['k+_cat11']
    k_cat12 = (par['k-_cat12'] + e / par['n_cat']) / par['k+_cat12']
    k_cat21 = (par['k-_cat21'] + e / par['n_cat']) / par['k+_cat21']
    k_cat22 = (par['k-_cat22'] + e / par['n_cat']) / par['k+_cat22']
    # get total CAT mRNA concentration divided by RNA-ribosome dissociation constants
    mcat_div_kcat_total = m_cat11 / k_cat11 + m_cat12 / k_cat12 + m_cat21 / k_cat21 + m_cat22 / k_cat22

    # RETURN THE ODE
    return [# mRNAs
            par['func_tog11'] * l * F[name2pos['F_tog11']] * par['c_tog11'] * par['a_tog11'] - (par['b_tog11'] + l) * x[name2pos['m_tog11']],
            par['func_tog12'] * l * F[name2pos['F_tog12']] * par['c_tog12'] * par['a_tog12'] - (par['b_tog12'] + l) * x[name2pos['m_tog12']],
            par['func_tog21'] * l * F[name2pos['F_tog21']] * par['c_tog21'] * par['a_tog21'] - (par['b_tog21'] + l) * x[name2pos['m_tog21']],
            par['func_tog22'] * l * F[name2pos['F_tog22']] * par['c_tog22'] * par['a_tog22'] - (par['b_tog22'] + l) * x[name2pos['m_tog22']],
            0,  # CAT expressed from the same operons as the toggle protein, not its own gene
            # proteins
            (e / par['n_tog11']) * (x[name2pos['m_tog11']] / k_het[name2pos['k_tog11']] / D) * R - (l + par['d_tog11']*p_prot) * x[name2pos['p_tog11']],
            (e / par['n_tog12']) * (x[name2pos['m_tog12']] / k_het[name2pos['k_tog12']] / D) * R - (l + par['d_tog12']*p_prot) * x[name2pos['p_tog12']],
            (e / par['n_tog21']) * (x[name2pos['m_tog21']] / k_het[name2pos['k_tog21']] / D) * R - (l + par['d_tog21']*p_prot) * x[name2pos['p_tog21']],
            (e / par['n_tog22']) * (x[name2pos['m_tog22']] / k_het[name2pos['k_tog22']] / D) * R - (l + par['d_tog22']*p_prot) * x[name2pos['p_tog22']],
            (e / par['n_cat']) * (mcat_div_kcat_total / D) * R - (l + par['d_cat']*p_prot) * x[name2pos['p_cat']],
    ]

# effective mRNA concentrations divided by ribosome-mRNA dissoc. consts,
# corrected for genes expressed from the same operon with some others
def twotoggles_add_eff_m_het_div_k_het(x, par, name2pos, num_circuit_genes,
                                    e, k_het):
    # calculate the effective mRNA concentrations divided by the ribosome-mRNA dissociation constants, no correction
    m_het_div_k_het_uncorr = jnp.sum(x[8:8 + num_circuit_genes] / k_het)

    # return the corrected value: here, add the values for the CAT gene co-expressed with toggle genes
    m_cat11 = x[name2pos['m_tog11']] * par['n_cat'] / par['n_tog11']
    m_cat12 = x[name2pos['m_tog12']] * par['n_cat'] / par['n_tog12']
    m_cat21 = x[name2pos['m_tog21']] * par['n_cat'] / par['n_tog21']
    m_cat22 = x[name2pos['m_tog22']] * par['n_cat'] / par['n_tog22']
    k_cat11 = (par['k-_cat11'] + e / par['n_cat']) / par['k+_cat11']
    k_cat12 = (par['k-_cat12'] + e / par['n_cat']) / par['k+_cat12']
    k_cat21 = (par['k-_cat21'] + e / par['n_cat']) / par['k+_cat21']
    k_cat22 = (par['k-_cat22'] + e / par['n_cat']) / par['k+_cat22']
    mcat_div_kcat_total = m_cat11 / k_cat11 + m_cat12 / k_cat12 + m_cat21 / k_cat21 + m_cat22 / k_cat22
    return m_het_div_k_het_uncorr + mcat_div_kcat_total
