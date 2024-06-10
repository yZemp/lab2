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


R = 1_000
def model_2(t, V0, C):
    tau = R * C
    return V0 * (np.exp(- t / tau))



###########################################################
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m

def interp_2(x, y, yerr, func = model_2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, .0000002)
    m.migrad()
    m.hesse()
    return m



#####################################################################
# Runtime

def main():

    cut_start = 400
    cut_end = 600
    xfull = clear_arr(data["x"].to_numpy())[cut_start:-cut_end]
    yfull = clear_arr(data["y"].to_numpy())[cut_start:-cut_end]

    N = int((max(yfull) - min(yfull)) / .08)
    print("N: ", N)
    aleph = 500 # Subsamples

    chi2s = []

    Cs = []
    Cerrs = []

    random.seed(0.)

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
        # yerr = (np.ones_like(y) * .08) / np.sqrt(12) # Non viene

        # m1 = interp(x, y, yerr)
        m1 = interp_2(x, y, yerr)

        if not i:
            print("----------------------------------------------- M1 -----------------------------------------------")
            print(m1.migrad())
            print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

            plt.errorbar(x, y, yerr, label = "Data (Sub sample)", linestyle = "", marker = "o", markersize = 1, c = "#053101", alpha = .8)
            lnsp = np.linspace(0, .000_16, 10_000)
            # plt.plot(lnsp, model(lnsp, *m1.values), label = "$V(t) = V_0 (1 - e^{-t/\\tau})$", c = "#a515d5")
            plt.plot(lnsp, model_2(lnsp, *m1.values), label = "$V(t) = V_0 (e^{- t / R C})$", c = "#a515d5")

            plt.xlabel("Tempo [s]")
            plt.ylabel("Tensione [V]")

            plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
            # plt.plot([], [], ' ', label = f"$\\C$ = ({param:.1f} $\pm$ {param_err:.1f}) s")

            plt.legend()
            plt.show()

        # chi2s.append(m1.fval / m1.ndof)
        chi2s.append(1. - chi2.cdf(m1.fval, df = m1.ndof))
        if m1.valid and 1. - chi2.cdf(m1.fval, df = m1.ndof) > .05:
            Cs.append(m1.values["C"])
            Cerrs.append(m1.errors["C"])

        if not i % 100:
            print(i)
            print("Chi2 ridotto:\t", chi2s[i])

    plt.hist(chi2s, sturges(chi2s), density = False)
    # plt.xlim(.100901659020, .100901659025224)
    plt.show()

    # Weighted average of extracted parameter
    avg = sum([Cs[i] * (1 / np.power(Cerrs[i], 2)) for i in range(len(Cs))]) / sum([(1 / np.power(Cerrs[i], 2)) for i in range(len(Cs))])
    err = np.sqrt(1 / sum([(1 / np.power(Cerrs[i], 2)) for i in range(len(Cs))]))
    plt.errorbar(np.array(range(len(Cs))), Cs, Cerrs, marker = "x", color = "#010101", linestyle = "none", alpha = .3)
    plt.hlines([avg, avg - err, avg + err], - 10, len(Cs) + 10, label = f"$C = ({avg:.11f} \pm {err:.11f})$", linestyles = "dotted", colors = "#a00101")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
