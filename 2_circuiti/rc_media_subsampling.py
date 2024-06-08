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
sheet_name = "rc_media"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
print(data)

def clear_arr(arr):
    return arr[~np.isnan(arr)]


###########################################################
# models

def model(t, V0, tau):
    return V0 * (np.exp(- t / tau))


###########################################################
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m




#####################################################################
# Runtime

def main():

    cut_start = 400
    cut_end = 600
    x = clear_arr(data["x"].to_numpy())[cut_start:-cut_end]
    y = clear_arr(data["y"].to_numpy())[cut_start:-cut_end]
    yerr = (np.ones_like(y) * .08) / np.sqrt(12)

    plt.errorbar(x, y, yerr, label = "Data (full)", linestyle = "", marker = "o", markersize = 1, c = "#55d9a5", alpha = .5)
    plt.legend()
    plt.show()

    N = 86

    chi2s = []

    random.seed(0.)

    for i in range(1_000):

        # plt.errorbar(x, y, yerr, label = "Data (full)", linestyle = "", marker = "o", markersize = 1, c = "#55d9a5", alpha = .5)

        xy = zip(x, y)
        xy_subsample = random.sample(list(xy), N)
        x, y = np.array(list(zip(*xy_subsample)))
        yerr = (np.ones_like(y) * .08)
        # yerr = (np.ones_like(y) * .08) / np.sqrt(12) # NON VIENE

        # plt.errorbar(x, y, yerr, label = "Data (Sub sample)", linestyle = "", marker = "o", markersize = 1, c = "#053101", alpha = .8)

        # print("----------------------------------------------- M1 -----------------------------------------------")
        m1 = interp(x, y, yerr)
        # print(m1.migrad())
        # print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

        # lnsp = np.linspace(0, .000_18, 10_000)
        # plt.plot(lnsp, model(lnsp, *m1.values), label = "$V(t) = V_0 (e^{-t/\\tau})$", c = "#a515d5")

        # plt.xlabel("Tempo [s]")
        # plt.ylabel("Tensione [V]")

        # tau = m1.values[1] * 100_000
        # tauerr = m1.errors[1] * 100_000

        # plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
        # plt.plot([], [], ' ', label = f"$\\tau$ = ({tau:.3f} $\pm$ {tauerr:.3f})x10^{5} s")

        # plt.legend()
        # plt.show()

        # chi2s.append(m1.fval / m1.ndof)
        chi2s.append(1. - chi2.cdf(m1.fval, df = m1.ndof))

        if not i % 10:
            print(i)
            print(chi2s[i])

    plt.hist(chi2s, sturges(chi2s), density = False)
    # plt.xlim(.100901659020, .100901659025224)
    plt.show()
    

if __name__ == "__main__":
    main()
