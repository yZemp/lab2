import numpy as np
import random
import scipy as sp
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import pandas as pd
# import sys
# sys.path.append("../..")

# import randgen
# import my_stats
# import funclib

##########################################################3
# Vars

sheet_id = "1dRjk3ARX3-TDBIuWrPaqR57W10_ucC5kEYAa6Lzmtn0"
sheet_name = "reticolo"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
# print(data)

##########################################################3
# Data

Theta_inc = 0.08226075666
lambd = 0.0000006328
d = .001

Ns1 = [1, 2, 3]
Ns2 = [-5, -4, -3, -2, -1]
Ns = [-5, -4, -3, -2, -1, 1, 2, 3]
ThetaN1 = data["ThetaN1"].to_numpy()
ThetaN1 = ThetaN1[~np.isnan(ThetaN1)]
ThetaN2 = data["ThetaN2"].to_numpy()
ThetaN2 = ThetaN2[~np.isnan(ThetaN2)]
ThetaN2 = ThetaN2[::-1]

ThetaN = np.concatenate((ThetaN2, ThetaN1))
print(ThetaN1, ThetaN2, ThetaN)

errors = .0011

#####################################################################
# Functions

def model_1(N, param):
    return np.arccos(np.cos(Theta_inc) - (N * param * .000001) / d)


def model_2(ThetaN, param):
    return (d / param) * (np.cos(Theta_inc) - np.cos(ThetaN))


#####################################################################
# Interpolation
    
def interp_1(x, y, yerr, func = model_1):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, .6)
    m.migrad()
    m.hesse()
    return m
    
def interp_2(x, y, yerr, func = model_2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, .6)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp_1(Ns, ThetaN, errors)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    


    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp_2(ThetaN, Ns, errors)
    # print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    
    # plt.figure(0)

    lnsp = np.linspace(Ns[0] - 1, Ns[-1] + 1, 10_000)
    
    plt.axes(xlabel = "$N$", ylabel = "$\\theta_N$")
    plt.errorbar(Ns, ThetaN, errors, linestyle = "", label = "Data", marker = "o", c = "#090909")
    plt.plot(lnsp, model_1(lnsp, *m1.values), label = "$\\theta_N(N)$", c = "#9929a9")
    

    plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$\\lambda$ = ({m1.values[0]:.3f} $\pm$ {m1.errors[0]:.3f})$\\times10^{-6}$ m")

    # plt.legend()
    # plt.show()

    # plt.figure(1)

    # lnsp = np.linspace(ThetaN[0] - .1, ThetaN[-1] + .1, 10_000)

    # plt.axes(xlabel = "$\\theta_N$", ylabel = "$N$")
    # plt.errorbar(ThetaN, Ns, errors, linestyle = "", marker = "o", c = "#191934")
    # plt.plot(lnsp, model_2(lnsp, *m2.values), label = "Legge di Stokeazzo 2 - PARTE 3: il continuo")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
