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


from gen_dataset import gen_dataset
from utils import make_params
from loglikelihood import estim_ll

# Packages ____________________________________________________________________


def one_mc(theta, seed, theta0):
    g1 = make_params(theta)

    df = gen_dataset(g1, seed)

    res = estim_ll(theta0, df)

    return res


if __name__ == "__main__":
    # Pruebas
    theta=[6.0, 0.02]
    res = one_mc(theta, 123, theta)
    print(res.x)




