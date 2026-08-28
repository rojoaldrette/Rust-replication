# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/loglikelihood.py
# Goal:           Funciones auxiliares del proyecto
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________


# Packages ____________________________________________________________________


import jax.numpy as jnp
from bellman import make_v_0, make_v_1, solve_bellman
from utils import make_mileage_grid
from params import Params
