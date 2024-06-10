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
sheet_name = "rlc_critico_sperim"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
print(data)

def clear_arr(arr):
    return arr[~np.isnan(arr)]


###########################################################
# models

def model(t, A, B, L):
    R = 2_100
    gamma = R / (2 * L)
    return (A * t + B) * np.exp(- gamma * t)

def model_sottosmorzato(t, V0, gamma, omega, phi):
    omega_0 = np.sqrt(np.power(omega, 2) - np.power(gamma, 2))
    return V0 * np.exp(- gamma * t) * np.sin(omega_0 * t + phi)


###########################################################
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, -100, -1.5, 1) # DO NOT TOUCH !!!
    m = Minuit(my_cost, -3e3, -.5, 6e-2)
    m.limits["L"] = (0, +np.inf)
    m.limits["A"] = (-np.inf, 0)
    m.limits["B"] = (-np.inf, 0)
    m.migrad()
    m.hesse()
    return m

def interp_sottosmorzato(x, y, yerr, func = model_sottosmorzato):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 600, 1300, 15000, 3)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():

    cut_start = 100
    cut_end = 100
    xfull = clear_arr(data["x"].to_numpy())[cut_start:-cut_end]
    yfull = clear_arr(data["y"].to_numpy())[cut_start:-cut_end]
    yerr = (np.ones_like(yfull) * .08) / np.sqrt(12)

    N = int((max(yfull) - min(yfull)) / .08) * 2 # oscillations
    N = int((max(yfull) - min(yfull)) / .08)
    print("N: ", N)
    aleph = 500 # Subsamples

    chi2s = []

    Cs = []
    Cerrs = []

    Ls = []
    Lerrs = []

    random.seed(2.)

    for i in range(aleph):

        # plt.errorbar(x, y, yerr, label = "Data (full)", linestyle = "", marker = "o", markersize = 1, c = "#55d9a5", alpha = .5)
        

        xy = list(zip(xfull, yfull))
        # print(list(xy))
        xy_subsample = np.array(random.sample(list(xy), k = N))

        x, y = np.array(list(zip(*list(xy_subsample))))
        # x = [xy_subsample[i][0] for i in range(len(xy_subsample))]
        # y = [xy_subsample[i][1] for i in range(len(xy_subsample))]
        # plt.errorbar(x, y, 0, label = "Subsample test", linestyle = "", marker = "o", markersize = 1)
        # plt.legend()
        # plt.show()

        yerr = (np.ones_like(y) * .08)
        # yerr = (np.ones_like(y) * .08) / np.sqrt(12) # dovrebbe venire

        m1 = interp(x, y, yerr)

        if not i:
            print("----------------------------------------------- M1 -----------------------------------------------")
            print(m1.migrad())
            print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

            plt.errorbar(x, y, yerr, label = "Data (Sub sample)", linestyle = "", marker = "o", markersize = 1, c = "#053101", alpha = .8)
            lnsp = np.linspace(-.000_3, .000_8, 10_000)
            # plt.plot(lnsp, model(lnsp, *m1.values), label = "$V(t) = V_0 (1 - e^{-t/\\tau})$", c = "#a515d5")
            plt.plot(lnsp, model(lnsp, *m1.values), label = "$V(t)$", c = "#a515d5")
            # plt.plot(lnsp, model_2_exponly(lnsp, *m1.values), label = "$V(t)$", c = "#04a905")

            plt.ylim(-3, 1)
            plt.xlabel("Tempo [s]")
            plt.ylabel("Tensione [V]")

            plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
            # plt.plot([], [], ' ', label = f"$\\C$ = ({param:.1f} $\pm$ {param_err:.1f}) s")

            plt.legend()
            plt.show()

        # chi2s.append(m1.fval / m1.ndof)
        chi2s.append(1. - chi2.cdf(m1.fval, df = m1.ndof))
        if m1.valid and 1. - chi2.cdf(m1.fval, df = m1.ndof) > .05:

            Ls.append(m1.values["L"])
            Lerrs.append(m1.errors["L"])


        if not i % 100:
            print(i)
            print("Chi2 ridotto:\t", chi2s[i])

    plt.hist(chi2s, sturges(chi2s), density = False)
    # plt.xlim(.100901659020, .100901659025224)
    plt.show()

    # Weighted average of extracted parameter
    Lavg = sum([Ls[i] * (1 / np.power(Lerrs[i], 2)) for i in range(len(Ls))]) / sum([(1 / np.power(Lerrs[i], 2)) for i in range(len(Ls))])
    Lerr = np.sqrt(1 / sum([(1 / np.power(Lerrs[i], 2)) for i in range(len(Ls))]))

    plt.errorbar(np.array(range(len(Ls))), Ls, Lerrs, marker = "x", color = "#010101", linestyle = "none", alpha = .3)
    plt.hlines([Lavg, Lavg - Lerr, Lavg + Lerr], - 10, len(Ls) + 10, label = f"$L = ({Lavg * 100:.2f} \pm {Lerr * 100:.2f})x10^{-2}$", linestyles = "dotted", colors = "#a00101")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
