# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/main.py
# Goal:           Funciones auxiliares del proyecto
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________


# Packages ____________________________________________________________________

import numpy as np
import pandas as pd
import jax.numpy as jnp
from dataclasses import replace
from params import Params
from probabilities import calc_probs
from gen_dataset import gen_dataset
from bellman import solve_bellman
from utils import make_params
from loglikelihood import estim_ll


# Main functioon ____________________________________________________________________


# Parameters ###################################

# Escenario base
g = Params()
# Diferent parameter
#g1 = replace(g, replacement_cost=4)


# Get bellman FP ##############################
V = solve_bellman(g)

# Get probabilities ##############################
p1, p0 = calc_probs(g)


# Generate dataset #############################

df = gen_dataset(g, 123)


# Estimation ################################




