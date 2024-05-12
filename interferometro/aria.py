import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

import sys
sys.path.append("/home/yzemp/Documents/Programming/lab2")
# import error_prop_bolde as errpropb

##########################################################3
# vars

# data = pd.read_excel("https://docs.google.com/spreadsheets/d/1bjtqJHRvWQMS7QxDNb8iBOs0CKm75ioh5B47gZAaU2Y/export?format=xlsx", sheet_name = None)
# print(data["gradi"])

d = 0.03
lam = 0.0000006328
# lam = 0.000000634
P0 = 101325 # Pascal

sheet_id = "1dRjk3ARX3-TDBIuWrPaqR57W10_ucC5kEYAa6Lzmtn0"
sheet_name = "vetro"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

DNs = [14.5, 10.5, 16.5, 6.3]
sigmaDNs = [.4,.4,.4,.3]
deltaps = [68, 50, 80, 30]

##########################################################
# models

def model(deltap, m):
    return ((2 * d * m) / (lam)) * deltap


##########################################################
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(deltaps, DNs, sigmaDNs)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

    plt.xlabel("$\Delta P$")
    plt.ylabel("$\Delta N$")

    plt.errorbar(deltaps, DNs, sigmaDNs, label = "Data", linestyle = "", marker = "o", c = "#151515")

    lnsp = np.linspace(min(deltaps) - 10, max(deltaps) + 10, 10_000)
    plt.plot(lnsp, model(lnsp, *m1.values), c = "#950ccc")

    m = float(f"{m1.values[0]:.8f}") * 10 ** 6
    sm = float(f"{m1.errors[0]:.8f}") * 10 ** 6

    plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"m = ({m} $\pm$ {sm})$\\times10^{-6}$ [$Pa^{-1}$]")
    plt.plot([], [], ' ', label = f"n = {m * P0 * 10 ** (- 6) + 1:.3f} $\\pm$ {sm * P0 * 10 ** (-6):.3f}")



    plt.legend(loc = 2)
    plt.show()

    

if __name__ == "__main__":
    main()
