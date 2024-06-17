'''
SWITCHING_TIME_TOOLS.PY - toolkit for the estimation of the Punisher's switching times
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

# SWITCHING TIME ESTIMATION FUNCTIONS ----------------------------------------------------------------------------------
# calculate the maximum likelihood estimator of switching time
def find_mle(switch_times,  # switching times
             n,  # number of samples
             T  # simulation time
             ):
    # don't have Type II censoring => 'prematurely finish' only if all trajectories switched
    r=n

    # find which trajectories didn't switch actually (zero switch time returned)
    no_switch = np.nonzero(switch_times==0)[0]

    # find which trajectories switched
    switched = np.nonzero(switch_times!=0)[0]
    d=len(switched) # number of switched trajectories

    # find MLE depending on whether all trajectories switched or not
    if(d>0): # some trajectories switched
        mle = np.sum(switch_times)/d + (n-d)*T/d # MLE is the average of switch times + average time for non-switched
    else:   # no trajectories switched
        mle=np.inf
        print('No trajectories switched. MLE undefined; assumed infinite.')

    return mle

# CONFIDENCE INTERVAL CALCULATION FUNCTIONS ----------------------------------------------------------------------------
# @jax.jit
def find_confint(mle,  # maximum likelihood estimator
                 n,  # number of samples
                 T,  # simulation time
                 alpha  # significance level
                 ):
    # find z(alpha/2) - quantile of standard normal distribution
    z_halfalpha = jax.scipy.stats.norm.ppf(alpha/2)

    # define optimisation problems
    opt_fun_xi_eq_z = jax.jit(lambda switch_time: xi(switch_time, mle, n, T)-z_halfalpha) # xi=z(alpha/2)
    opt_fun_xi_eq_minusz = jax.jit(lambda switch_time: xi(switch_time, mle, n, T)+z_halfalpha) # xi=-z(alpha/2)

    # find left border of the confidence interval
    xi_eq_minusz_problem=jaxopt.Bisection(optimality_fun=opt_fun_xi_eq_minusz,
                     lower=1e-5, upper=1e5,
                     maxiter=10000, tol=1e-18,
                     check_bracket=False)
    border_left = xi_eq_minusz_problem.run().params

    # find one side of the confidence interval
    xi_eq_z_problem = jaxopt.Bisection(optimality_fun=opt_fun_xi_eq_z,
                                       lower=1e-5, upper=1e5,
                                       maxiter=10000, tol=1e-18,
                                       check_bracket=False)
    border_right = xi_eq_z_problem.run().params

    # return
    return border_left, border_right


def xi(switch_time,  # supposed switching time
       mle,  # maximum likelihood estimator
       n,  # number of samples
       T  # simulation time
       ):
    # calculate switching probabilities
    Q=jnp.exp(-T/switch_time) # probability of not switching by time T
    P=1-Q # probability of switching by time T

    # calculate auxiliary variables
    u=(mle-switch_time)/switch_time
    d1=Q*(-T/switch_time)/P
    d2=Q

    # return xi
    return u * jnp.sqrt(n*P) / jnp.sqrt(1-2*d1*u+d2*jnp.square(u))


# SWITCHING TIME ESTIMATION BASED ON NUMBER OF SWTICHINGS ALONE
def mle_numswitch_alone(switch_times,  # switching times
             n,  # number of samples
             T  # simulation time
             ):
    # find which trajectories switched
    switched = np.nonzero(switch_times!=0)[0]
    d=len(switched) # number of switched trajectories

    # find MLE depending on whether all trajectories switched or not
    if(d>0): # some trajectories switched
        mle = -T/np.log(1-d/n)
    else:   # no trajectories switched
        mle=np.inf
        print('No trajectories switched. MLE undefined; assumed infinite.')

    return mle