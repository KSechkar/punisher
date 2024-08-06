'''
DESIGN_GUIDANCE_TOOLS.PY - toolkit for the design of synthetic gene circuits
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
import pickle
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, palettes as bkpalettes, transform as bktransform
from math import pi
import time

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
import synthetic_circuits as circuits
from cell_model import *

# F_SWITCH FUNCTION VALUES ---------------------------------------------------------------------------------------------
# real value
def F_real_calc(p_switch, par):
    p_switch_dependent_term = (p_switch * par['p_switch_ac_frac'] / par['K_switch']) ** par['eta_switch']
    return par['baseline_switch'] + (1 - par['baseline_switch']) * (
            p_switch_dependent_term / (p_switch_dependent_term + 1))


# required value
def F_req_calc(p_switch, xi, par, cellvars):
    return p_switch * (1 + cellvars['chi_switch']) * \
        xi / (cellvars['xi_switch_max'] + cellvars['xi_int_max']) / \
        (par['M'] * (1 - par['phi_q']) / par['n_switch'] * cellvars['xi_switch_max'] / (
                cellvars['xi_switch_max'] + cellvars['xi_int_max']) - p_switch * (1 + cellvars['chi_switch']))


# F_SWITCH FUNCTION GRADIENTS ------------------------------------------------------------------------------------------
# real value
def dFreal_dpswitch_calc(p_switch, par):
    K_div_ac_frac = par['K_switch'] / par['p_switch_ac_frac']
    return par['eta_switch'] * (1 - par['baseline_switch']) * K_div_ac_frac ** par[
        'eta_switch'] * p_switch ** (par['eta_switch'] - 1) / \
        (K_div_ac_frac ** par['eta_switch'] + p_switch ** par['eta_switch']) ** 2


# required value
def dFreq_dpswitch_calc(p_switch, xi, par, cellvars):
    MQns = par['M'] * (1 - par['phi_q']) / par['n_switch']
    return xi * (1+cellvars['chi_switch']) * cellvars['xi_switch_max'] * MQns / (
            cellvars['xi_switch_max'] * MQns -
            p_switch * (1+cellvars['chi_switch']) * (cellvars['xi_switch_max'] + cellvars['xi_int_max'])
    ) ** 2

# FINDING THE THRESHOLD BIFURCATION: PARAMETRIC APPROACH ---------------------------------------------------------------
# difference of gradients at the fixed point for a given value of F_switch
def gradiff_from_F(F, par, cellvars):
    # reconstruct p_switch and xi
    p_switch = pswitch_from_F(F, par)
    xi = xi_from_F_and_pswitch(F, p_switch, par, cellvars)

    # get the gradients
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, xi, par, cellvars)
    return dFreal_dpswitch - dFreq_dpswitch


# difference of gradients at the fixed point for a given value of p_switch
def gradiff_from_pswitch(p_switch, par, cellvars):
    # reconstruct F_switch and xi
    F = F_real_calc(p_switch, par)
    xi = xi_from_F_and_pswitch(F, p_switch, par, cellvars)

    # get the gradients
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, xi, par, cellvars)
    return dFreal_dpswitch - dFreq_dpswitch


# find p_switch value for a given real value of F_switch
def pswitch_from_F(F, par):
    return (par['K_switch'] / par['p_switch_ac_frac']) * \
        ((F - par['baseline_switch']) / (1 - F)) ** (1 / par['eta_switch'])


# find the value of xi yielding a fixed point for given F_switch and p_switch values
def xi_from_F_and_pswitch(F, p_switch, par, cellvars):
    return F * (cellvars['xi_switch_max'] + cellvars['xi_int_max']) / (p_switch * (1 + cellvars['chi_switch'])) * \
        (par['M'] * (1 - par['phi_q']) / par['n_switch'] * cellvars['xi_switch_max'] / (
                cellvars['xi_switch_max'] + cellvars['xi_int_max']) - p_switch * (1 + cellvars['chi_switch']))


# just for convenience, can get a pair of xi and p_switch values
def pswitch_and_xi_from_F(F, par, cellvars):
    p_switch = pswitch_from_F(F, par)
    xi = xi_from_F_and_pswitch(F, p_switch, par, cellvars)
    return jnp.array([p_switch,xi])


# upper bound for xi to find the saddle point at the bifurcation (inflexion in real F_switch values)
def pswitch_inflexion_in_Freal(par):
    return ((par['eta_switch'] - 1) / (par['eta_switch'] + 1)) ** (1 / par['eta_switch']) * (
                par['K_switch'] / par['p_switch_ac_frac'])


# FINDING THE GUARANTEED CHANGE IN P_INT EXPRESSION --------------------------------------------------------------------
# find the difference of values between the real and required F_switch values
# (used for optimisation in terms of p_switch)
def validff_to_find_pswitch(p_switch,
                            xi,
                            par, cellvars):
    return F_real_calc(p_switch, par) - F_req_calc(p_switch, xi, par, cellvars)


# upper bound for p_switch to find the non-saddle fixed point - for a given (threshold) value of xi
def pswitch_upper_bound_4nonsaddle(xi,
                                   par, cellvars):
    return 1 / (1 + cellvars['chi_switch']) * \
        par['M'] * (1 - par['phi_q']) / par['n_switch'] * \
        cellvars['xi_switch_max'] / (cellvars['xi_switch_max'] + cellvars['xi_int_max'] + xi)


# find the integrase protein concentration for a given value of p_switch and burden xi
def pint_from_pswitch_and_xi(p_switch,
                             xi,
                             par, cellvars):
    # get the F value
    F = F_real_calc(p_switch, par)

    return 1 / (1 + cellvars['chi_int']) * \
        par['M'] * (1 - par['phi_q']) / par['n_switch'] * \
        (F * cellvars['xi_int_max']) / (cellvars['xi_switch_max'] + cellvars['xi_int_max'] + xi)


# find integrase activity for a given p_int value
def intact_from_pint(p_int, par):
    return (p_int/par['K_bI~'])**4/(1+(p_int/par['K_bI~'])**4)


# AUXILIARY FINCTIONS FOR DIAGNOSTICS AND CASE INVESTIGATION -----------------------------------------------------------
# Return squared difference between the real and required F_switch values and their gradients
def differences_squared(pswitch_xi,# decision variables: p_switch and xi
                        par,  # dictionary with model parameters
                        cellvars  # cellular variables that we assume to be constant
                        ):
    # unpack the decision variables
    p_switch = pswitch_xi[0]
    xi = pswitch_xi[1]

    # find the values of F_switch functions and their gradients
    F_real = F_real_calc(p_switch, par)
    F_req = F_req_calc(p_switch, xi, par, cellvars)
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, xi, par, cellvars)

    return jnp.array([(F_real - F_req) ** 2, (dFreal_dpswitch - dFreq_dpswitch) ** 2])

# CHECK IF THE THRESHOLD/BIFURCATION POINT EXISTS ----------------------------------------------------------------------
# to do so, it is enough to check the sign of the difference of gradients in the inflexion point of real F_switch values
def check_if_threshold_exists(par, cellvars):
    p_switch_inflexion = pswitch_inflexion_in_Freal(par) # get p_switch value in the inflexion point
    return gradiff_from_pswitch(p_switch_inflexion, par, cellvars) > 0 # TRUE <=> threshold exists


# FIND THRESHOLD AND GUARANTEED FOLD EXPRESSION CHANGES ----------------------------------------------------------------
# find the threshold and guaranteed fold expression changes for a given set of parameters
def threshold_gfchanges(par, cellvars):
    pswitch_inflexion = pswitch_inflexion_in_Freal(
        par)  # upper bound of feasible region for p_switch (inflexion point in real F_switch)
    F_upper_bound = F_real_calc(pswitch_inflexion, par)  # upper bound of feasible region for F
    F_lower_bound = par['baseline_switch']  # lower bound of feasible region for F (corresponds to p_switch=0)

    # FIND THE THRESHOLD BIFURCATION POINT -----------------------------------------------------------------------------
    # create an instance of the optimisation problem
    threshold_problem = jaxopt.Bisection(optimality_fun=gradiff_from_F,
                                         lower=F_lower_bound, upper=F_upper_bound,
                                         maxiter=10000, tol=1e-18,
                                         check_bracket=False)  # required for vmapping and jitting

    # solve the optimisation problem
    F_threshold = threshold_problem.run(par=par, cellvars=cellvars).params
    # unpack the solution
    p_switch_threshold=pswitch_from_F(F_threshold,par)
    xi_threshold = xi_from_F_and_pswitch(F_threshold,p_switch_threshold,par,cellvars)


    # FIND THE GUARANTEED CHANGE IN INTEGRASE PROTEIN EXPRESSION -------------------------------------------------------
    # find the non-saddle fixed point for the threshold xi value
    p_switch_sup = pswitch_upper_bound_4nonsaddle(xi_threshold, par,
                                                  cellvars)  # supremum of biologically possible p_switch values for a given xi
    nonsaddle_problem = jaxopt.Bisection(optimality_fun=validff_to_find_pswitch,
                                         lower=pswitch_inflexion, upper=p_switch_sup,  # non-saddle f.p. is found between the inflexion and the supremum
                                         maxiter=10000, tol=1e-18,
                                         check_bracket=False)  # required for vmapping and jitting

    p_switch_nonsaddle = nonsaddle_problem.run(xi=xi_threshold, par=par,
                                                   cellvars=cellvars).params  # optimise to find the p_switch value at non-saddle fixed point

    # find the integrase protein expression at the saddle and non-saddle fixed point
    p_int_saddle = pint_from_pswitch_and_xi(p_switch_threshold, xi_threshold, par, cellvars)
    p_int_nonsaddle = pint_from_pswitch_and_xi(p_switch_nonsaddle, xi_threshold, par, cellvars)

    # find the guaranteed fold-change in integrase protein expression
    gfchange_int = p_int_nonsaddle / p_int_saddle

    # find the guaranteed fold-change in F value (just as an extra)
    gfchange_F = F_real_calc(p_switch_nonsaddle, par) / F_threshold

    # find the guaranteed fold-change in integrase activity
    intact_saddle = (p_int_saddle/par['K_bI~'])**4/ (1+(p_int_saddle/par['K_bI~'])**4)
    intact_nonsaddle = (p_int_nonsaddle/par['K_bI~'])**4/ (1+(p_int_nonsaddle/par['K_bI~'])**4)
    gfchange_intact = intact_nonsaddle/intact_saddle

    return jnp.array([p_switch_threshold, xi_threshold, gfchange_int, gfchange_F, gfchange_intact])


# FIND THE PUNISHER'S BOUNDARY BETWEEN THE STABLE EQUILIBRIA'S BASIONS OF ATTRACTION -----------------------------------
# assuming burden is above the switching threshold
def find_basin_border(par, cellvars):

    # greatest possible p_switch level - for zero extra burden so that it's max across all conditions
    p_switch_upper_bound = jnp.ceil(cellvars['xi_switch_max'] * (1 / (1 + cellvars['chi_switch'])) / (
            cellvars['xi_switch_max'] + cellvars['xi_int_max'] + cellvars['xi_prot'] + cellvars['xi_cat'] + cellvars[
        'xi_a'] + cellvars['xi_r']
    ) * par['M'] * (1 - par['phi_q']) / par['n_switch'])  # upper bound for p_switch (to get the high equilibrium)

    # total burden
    xi_total = cellvars['xi_prot'] + cellvars['xi_cat'] + cellvars['xi_a'] + cellvars['xi_r'] + cellvars[
        'xi_other_genes']

    # PIN DOWN THE EQUILBIRA TO THE NEAREST INTEGER --------------------------------------------------------------------
    # get the axis of integer p_switch values
    p_switch_integers = jnp.arange(0, p_switch_upper_bound + 0.5, 1)

    # find the squared differences between the required and real F values
    sqdiffs = (F_real_calc(p_switch_integers, par) - F_req_calc(p_switch_integers, xi_total, par, cellvars)) ** 2
    previous_sqdiff_geq = jnp.concatenate((jnp.array([False]), sqdiffs[:-1] >= sqdiffs[1:]))
    next_sqdiff_geq = jnp.concatenate((sqdiffs[:-1] <= sqdiffs[1:], jnp.array([False])))

    equilibria = p_switch_integers[jnp.logical_and(previous_sqdiff_geq, next_sqdiff_geq)]
    unstable_eq_integer = equilibria[1]

    # LOOK FOR THE BOUNDARY (UNSTABLE EQUILIBRIUM) IN THE VICINITY OF THE INTEGER VALUE --------------------------------
    diff = lambda p_switch: F_real_calc(p_switch, par) - F_req_calc(p_switch, xi_total, par, cellvars) # IMPORTANT: NOT A SQUARE TO HAVE DIFFERENT SIGNS EITHER SIDE OF THE UNSTABLE EQUILIBRIUM
    unsteq_problem = jaxopt.Bisection(optimality_fun=diff,
                     lower=float(unstable_eq_integer)-2, upper=float(unstable_eq_integer)+2,
                     maxiter=10000, tol=1e-18,
                     check_bracket=False)
    border = unsteq_problem.run().params

    return border
