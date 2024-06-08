import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import chi2, norm, cauchy
import pandas as pd
from collections import defaultdict

import sys
sys.path.append("/home/yzemp/Documents/Programming/lab2")
from my_stats import stat, sturges


###########################################################
# vars

sheet_id = "1DR5TWcdKj22btlrAdPJKfSGy_9bbEQaM7yqhpW0MGiA"
sheet_name = "rl"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
# print(data)

def clear_arr(arr):
    return arr[~np.isnan(arr)]


###########################################################
# models

def model(t, V0, tau):
    return V0 * (1 - np.exp(- t / tau))


###########################################################
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m

#####################################################################
# Main

def main():

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(x, y, yerr)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    plt.errorbar(x, y, yerr, label = "Data", linestyle = "", marker = "o", markersize = 3, c = "#55d9a5", alpha = .4)
    lnsp = np.linspace(x[0], x[-1], 10_000)
    plt.plot(lnsp, model(lnsp, *m1.values), label = "$V(t) = V_0 (1 - e^{-t/\\tau})$", c = "#a515d5")

    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensione [V]")

    tau = m1.values[1] * 100_000
    tauerr = m1.errors[1] * 100_000

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$\\tau$ = ({tau:.3f} $\pm$ {tauerr:.3f})x10^{5} s")

    plt.legend()
    plt.show()


    print("-------------------------------------------- ERRORS --------------------------------------------")
    sqm = np.sqrt(np.sum(np.power((y - model(x , *m1.values)), 2)) / len(x))
    yerr_2 = np.ones_like(x) * sqm
    # print(sqm)


    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp(x, y, yerr_2)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    
    plt.errorbar(x, y, yerr_2, label = "Data", linestyle = "", marker = "o", markersize = 3, c = "#55d9a5", alpha = .4)
    lnsp = np.linspace(x[0], x[-1], 10_000)
    plt.plot(lnsp, model(lnsp, *m2.values), label = "$V(t) = V_0 (1 - e^{-t/\\tau})$", c = "#a515d5")

    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensione [V]")

    tau = m2.values[1] * 100_000
    tauerr = m2.errors[1] * 100_000

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m2.fval / m2.ndof):.3f}, P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$\\tau$ = ({tau:.3f} $\pm$ {tauerr:.3f})x10^{5} s")

    plt.legend()
    plt.show()



###########################################################
# Error analysis and data manipulation


# Raw data
ystep = .08

cut_start = 500
cut_end = 2100
x = clear_arr(data["x"].to_numpy())[cut_start:cut_end]
y = clear_arr(data["y"].to_numpy())[cut_start:cut_end]
error = 80 / len(x)
error = .08 / np.sqrt(12)
yerr = (np.ones_like(y) * ystep) / np.sqrt(12)
yerr = (np.ones_like(y) * error)

# Actual interpolation of data
if __name__ == "__main__":
    main()