import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd
import random

import sys
sys.path.append("/home/yzemp/Documents/Programming/lab2")

from my_stats import sturges

###########################################################
# vars

sheet_id = "1DR5TWcdKj22btlrAdPJKfSGy_9bbEQaM7yqhpW0MGiA"
sheet_name = "rlc_sott_media"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
print(data)

def clear_arr(arr):
    return arr[~np.isnan(arr)]


###########################################################
# models

# def model(t, V0, gamma, omega_0, phi):
#     return V0 * np.exp(- gamma * t) * np.sin(omega_ * t + phi)

def model_ext(t, V0, gamma, omega_0, phi):
    omega = np.sqrt(np.power(omega_0, 2) - np.power(gamma, 2))
    return V0 * np.exp(- gamma * t) * np.sin(omega * t + phi)

def model_2(t, V0, L, C, phi):
    R = 100
    gamma = R / (2 * L)
    omega_0 = 1 / (L * C)
    omega = np.sqrt(omega_0 - np.power(gamma, 2))
    return V0 * np.exp(- gamma * t) * np.sin(omega * t + phi)

def model_2_exponly(t, V0, L, C, phi):
    R = 100
    gamma = R / (2 * L)
    omega_0 = 1 / (L * C)
    omega = np.sqrt(omega_0 - np.power(gamma, 2))
    return V0 * np.exp(- gamma * t)


###########################################################
# interpolations

def interp(x, y, yerr, func = model_ext):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 600, 1300, 15000, 3)
    m.limits["V0"] = (0, +np.inf)
    m.limits["gamma"] = (0, +np.inf)
    m.limits["omega_0"] = (0, +np.inf)
    m.migrad()
    m.hesse()
    return m

def interp_2(x, y, yerr, func = model_2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    # m = Minuit(my_cost, 10, .0817, 102.5e-9, 1)
    m = Minuit(my_cost, 100, 6.8e-3, 90e-9, 1) # DO NOT TOUCH !!!
    # m = Minuit(my_cost, 100, 6.8e-3, 100e-9, 1)
    m.limits["V0"] = (0, +np.inf)
    m.limits["L"] = (0, +np.inf)
    m.limits["C"] = (0, +np.inf)
    m.migrad()
    m.hesse()
    return m

#####################################################################
# Runtime

def main():

    cut_start = 100
    cut_end = 100
    x = clear_arr(data["x"].to_numpy())[cut_start:-cut_end]
    y = clear_arr(data["y"].to_numpy())[cut_start:-cut_end]
    yerr = (np.ones_like(y) * .08) / np.sqrt(12)

    # plt.errorbar(x, y, yerr, label = "Data (full)", linestyle = "", marker = "o", markersize = 1, c = "#55d9a5", alpha = .5)
    # plt.legend()
    # plt.show()

    N = int((max(y) - min(y)) / .08) * 10 # oscillations
    print("N: ", N)

    chi2s = []

    random.seed(0.)

    for i in range(1_000):

        # plt.errorbar(x, y, yerr, label = "Data (full)", linestyle = "", marker = "o", markersize = 1, c = "#55d9a5", alpha = .5)

        xy = zip(x, y)
        xy_subsample = random.sample(list(xy), N)
        x, y = np.array(list(zip(*xy_subsample)))
        yerr = (np.ones_like(y) * .08)
        yerr = (np.ones_like(y) * .08) / np.sqrt(12) # VIENE DI BRUTTO


        # m1 = interp(x, y, yerr)
        m1 = interp_2(x, y, yerr)

        if not i:
            print("----------------------------------------------- M1 -----------------------------------------------")
            print(m1.migrad())
            print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

            plt.errorbar(x, y, yerr, label = "Data (Sub sample)", linestyle = "", marker = "o", markersize = 1, c = "#053101", alpha = .8)
            lnsp = np.linspace(0, .003, 10_000)
            # plt.plot(lnsp, model_ext(lnsp, *m1.values), label = "$V(t)$", c = "#a515d5")
            plt.plot(lnsp, model_2(lnsp, *m1.values), label = "$V(t)$", c = "#a515d5")
            plt.plot(lnsp, model_2_exponly(lnsp, *m1.values), label = "$V(t)$", c = "#04a905")

            plt.xlabel("Tempo [s]")
            plt.ylabel("Tensione [V]")

            tau = m1.values[1] * 100_000
            tauerr = m1.errors[1] * 100_000

            plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
            plt.plot([], [], ' ', label = f"$\\tau$ = ({tau:.3f} $\pm$ {tauerr:.3f})x10^{5} s")

            plt.legend()
            plt.show()

        # chi2s.append(m1.fval / m1.ndof)
        chi2s.append(1. - chi2.cdf(m1.fval, df = m1.ndof))

        if not i % 100:
            print(i)
            print(chi2s[i])

    plt.hist(chi2s, sturges(chi2s), density = False)
    # plt.xlim(.100901659020, .100901659025224)
    plt.show()
    

if __name__ == "__main__":
    main()
