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

import pandas as pd
import jax.numpy as jnp
from gen_dataset import gen_dataset
from dataclasses import replace
from params import Params
from probabilities import calc_probs
import jax
from scipy.optimize import minimize
import numpy as np
import time


# The meat _____________________________________________________________________

def treat_data(cell_df):
    mileage = jnp.asarray(cell_df["mileage"].values)
    n = jnp.asarray(cell_df["frequency"].values)
    r = jnp.asarray(cell_df["replace"].values)

    return mileage, n, r


def make_params(theta):
    g = Params()
    g1 = replace(g, replacement_cost=theta[0], mileage_cost=theta[1])
    return g1


def log_lh(theta, mileage, n, r):

    p1, _ = calc_probs(theta)

    p = p1[mileage]
    eps = 1e-10
    p = jnp.clip(p, eps, 1 - eps)

    ll = jnp.sum(
        r * jnp.log(p)
        + (n - r) * jnp.log1p(-p)
    )

    return ll


@jax.jit
def objective(theta, mileage, n, r):

    theta1 = make_params(theta)

    return -log_lh(theta1, mileage, n, r)


def wrapper(theta0, data):

    mileage, n, r = treat_data(data)

    return objective(theta0, mileage, n, r)





if __name__ == "__main__":

    df = pd.read_csv('output/dataset_cb_example.csv')

    m, n, r = treat_data(df)

    theta = [1.0, 1.0]

    #print(wrapper(theta, df))

    diff = jax.grad(wrapper)

    #print(diff(theta, df))

    start_time = time.perf_counter()

    result = minimize(
        fun=wrapper,
        x0=np.array([8.0, 0.09]),
        jac=diff,
        args=(df),
        method="L-BFGS-B",
        bounds=[
            (1e-6, None), 
            (1e-6, None) 
        ]
        )
    end_time = time.perf_counter()
    print(result)

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")








