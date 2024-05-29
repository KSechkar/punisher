'''
GET_STEADY_STATE.PY - get the system's steady state and steady-state vaalues of important variables
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
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

# SIMULATE THE SYSTEM FOR 50 H TO GET ITS STEADY STATE [vmappable]------------------------------------------------------
