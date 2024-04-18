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


#####################################################################
# Data

sheet_id = "1dRjk3ARX3-TDBIuWrPaqR57W10_ucC5kEYAa6Lzmtn0"
sheet_name = "Foglio1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

D = (170.1 - 1.8) * 1e-2
lam = 632.8 * 1e-9
sigmar = 3 * 1e-3

# print(D, lam)

# raggi1 = (data["Diametro1 [cm]"].to_numpy() * 1e-2) / 2
raggi1 = (np.asarray([2.8,4.3,5.4,6.3,7.1,7.8,8.5,9.1,9.6]) * 1e-2) / 2
errs1 = [abs((r * D) / np.power(np.power(r, 2) + np.power(D, 2), 1.5) * sigmar)for r in raggi1]
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y1 = [np.cos(np.arctan(r / D)) for r in raggi1]


# raggi2 = (data["Diametro2 [cm]"].to_numpy() * 1e-2) / 2
raggi2 = (np.asarray([2.1,4.0,5.1,6.1,6.9,7.7]) * 1e-2) / 2
errs2 = [abs((r * D) / np.power(np.power(r, 2) + np.power(D, 2), 1.5) * sigmar)for r in raggi2]
x2 = [1, 2, 3, 4, 5, 6]
y2 = [np.cos(np.arctan(r / D)) for r in raggi2]

# raggi3 = (data["Diametro3 [cm]"].to_numpy() * 1e-2) / 2
raggi3 = (np.asarray([1.6,3.7,4.9,5.9,6.9,7.6,8.2,8.9]) * 1e-2) / 2
errs3 = [abs((r * D) / np.power(np.power(r, 2) + np.power(D, 2), 1.5) * sigmar)for r in raggi3]
x3 = [1, 2, 3, 4, 5, 6, 7, 8]
y3 = [np.cos(np.arctan(r / D)) for r in raggi3]



#####################################################################
# Functions

def model_linear(x, d, B):
    return  (lam / (2 * d)) * x + B

#####################################################################
# Interpolation
    
def interp_linear(x, y, yerr, func = model_linear):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, - .01, 1)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():
    # print(f"Raggi: {raggi1}\nX1: {x1}\n Y1 = cos(theta): {y1}")

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp_linear(x1, y1, errs1)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

    print(m1.values["d"])


    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp_linear(x2, y2, errs2)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")

    print(m2.values["d"])


    print("----------------------------------------------- M3 -----------------------------------------------")
    m3 = interp_linear(x3, y3, errs3)
    print(m3.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")

    print(m3.values["d"])





    plt.errorbar(x1, y1, errs1, linestyle = "", c = "red", marker = "o")
    plt.errorbar(x2, y2, errs2, linestyle = "", c = "green", marker = "o")
    plt.errorbar(x3, y3, errs3, linestyle = "", c = "blue", marker = "o")
    
    lnsp = np.linspace(0, 10, 10_000)
    plt.plot(lnsp, model_linear(lnsp, * m1.values), c = "red", label = "Legge di stokeazzo 1")
    plt.plot(lnsp, model_linear(lnsp, * m2.values), c = "green",  label = "Legge di stokeazzo 2")
    plt.plot(lnsp, model_linear(lnsp, * m3.values), c = "blue",  label = "Legge di stokeazzo 3")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
