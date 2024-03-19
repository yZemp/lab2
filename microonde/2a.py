import numpy as np
import random
import scipy as sp
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
# import sys
# sys.path.append("../..")

# import randgen
# import my_stats
# import funclib

#####################################################################
# Data

arrthetastar = [45,47,48,49,50,51,55] # FILL
arrthetastar_rad = [(t * 2 * np.pi) / 360 for t in arrthetastar]

arrs = [4.2,4.28,4.31,4.32,4.29,4.23,4] # FILL
serrors = np.ones_like(arrs) * 0.09 # TODO


#####################################################################
# Functions

def quad_model(x, A, B, C):
    return np.power(x, 2) * A + B * x + C

def trick_quad_model(x, A, xv, yv):
    return A * np.power(x - xv, 2) + yv

def scatter(x, y, yerr):
    plt.errorbar(x, y, yerr)
    plt.show()


#####################################################################
# Interpolation

def quad_interp(x, y, yerr, func = quad_model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1, 1)
    m.migrad()
    m.hesse()
    return m

def trick_quad_interp(x, y, yerr, func = trick_quad_model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, -1, 1, 1)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():
    # scatter(arrthetastar_rad, arrs, serrors)

    # print("----------------------------------------------- M1 -----------------------------------------------")
    # m1 = quad_interp(arrthetastar_rad, arrs, serrors)
    # print(m1.migrad())
    # print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    # xmax = - m1.values["B"] / (2 * m1.values["A"])
    # print(f"Theta (Rad) massimo:\t{xmax}")
    # print(f"Theta (°) massimo:\t{(360 * xmax) / (2 * np.pi)}")


    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = trick_quad_interp(arrthetastar_rad, arrs, serrors)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    xmax = m2.values["xv"]
    xmax_err = m2.errors["xv"]
    print(f"Theta (Rad) massimo:\t{xmax} +- {xmax_err}")
    print(f"Theta (°) massimo:\t{(360 * xmax) / (2 * np.pi)} +- {(360 * xmax_err) / (2 * np.pi)}")

    plt.axes(xlabel = "Theta* [Rad]", ylabel = "Segnale [V]")

    plt.errorbar(arrthetastar_rad, arrs, serrors, linestyle = "", c = "#0e0e0e", marker = "o")

    plt.vlines(xmax, 3.9, 4.5, linestyles = "dotted", label = f"Theta max (°) = {(360 * xmax) / (2 * np.pi):.1f} ± {(360 * xmax_err) / (2 * np.pi):.1f}")

    plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")

    lnsp = np.linspace(arrthetastar_rad[0] - .05, arrthetastar_rad[-1] + .01, 10_000)
    # plt.plot(lnsp, quad_model(lnsp, *m1.values), label = "Normale")
    plt.plot(lnsp, trick_quad_model(lnsp, *m2.values), label = "", c = "#049304")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
