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

d = 0.00575
lam = 0.0000006328

sheet_id = "1dRjk3ARX3-TDBIuWrPaqR57W10_ucC5kEYAa6Lzmtn0"
sheet_name = "vetro"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

frange = data["Frange"].to_numpy()
sigmaf = data["Sigma frange"].to_numpy()
angoli = data["Angolo"].to_numpy()
sigmaa = np.ones_like(angoli) * .1 * np.pi / 180

##########################################################3
# models

def model(theta, n):
    return (2 * d * (n - 1) * (1 - np.cos(theta))) / (lam * (n - 1 + np.cos(theta)))


##########################################################3
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

    plt.errorbar(angoli, frange, sigmaf, xerr = sigmaa, label = "Data (1)", linestyle = "", marker = "o", c = "#151515")

    plt.legend()
    plt.show()


    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(angoli, frange, sigmaf)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")


    # SECOND INTERPOLATION (WITH X ERRORS)
    for i in range(len(angoli)):
        yl = model(angoli[i] - sigmaa[i], *m1.values)
        yr = model(angoli[i] + sigmaa[i], *m1.values)
        diff = abs(yr - yl)
        sigmaf[i] = np.sqrt(np.power(sigmaf[i], 2) + np.power(diff, 2))

    
    plt.errorbar(angoli, frange, sigmaf, label = "Data (2)", linestyle = "", marker = "o", c = "#151515")

    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp(angoli, frange, sigmaf)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    
    lnsp = np.linspace(angoli[0] - 0.02, angoli[-1] + 0.02, 10_000)
    plt.plot(lnsp, model(lnsp, *m2.values), label = "Legge di stokeazzo THE RETURN", c = "#a515d5")

    plt.xlabel("$\Delta \\theta$")
    plt.ylabel("$\Delta N$")

    plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    plt.plot([], [], ' ', label = f"n = {m2.values[0]:.2f} $\pm$ {m2.errors[0]:.2f}")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
