# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/utils.py
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________

import jax.numpy as jnp


def make_mileage_grid(p):
    return jnp.arange(
        p.mileage_min,
        p.mileage_max + p.mileage_step,
        p.mileage_step
    )


