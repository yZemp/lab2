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

arrthetastar = [46,47,48,49,50,51,52,53,54] # FILL
arrthetastar = arrthetastar[2:-1]
arrthetastar_rad = [(t * 2 * np.pi) / 360 for t in arrthetastar]

arrs = [0.48,0.55,0.61,0.4,0.21,0.04,0.06,0.14,0.16] # FILL
arrs = arrs[2:-1]
serrors = np.ones_like(arrs) * 0.09 # TODO


#####################################################################
# Functions

def quad_model(x, A, B, C):
    return np.power(x, 2) * A + B * x + C

def trick_quad_model(x, A, xv, yv):
    return A * np.power(x - xv, 2) + yv

def trick_quad_model_V2(x, A, xv):
    return A * np.power(x - xv, 2)

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
    m = Minuit(my_cost, 1000, 1, 4)
    m.migrad()
    m.hesse()
    return m

def trick_quad_interp_V2(x, y, yerr, func = trick_quad_model_V2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, -1, 1)
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
    # xmin = - m1.values["B"] / (2 * m1.values["A"])
    # print(f"Theta (Rad) minimo:\t{xmin}")
    # print(f"Theta (°) minimo:\t{(360 * xmin) / (2 * np.pi)}")


    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = trick_quad_interp(arrthetastar_rad, arrs, serrors)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    xmin = m2.values["xv"]
    xmin_err = m2.errors["xv"]
    print(f"Theta (Rad) minimo:\t{xmin} +- {xmin_err}")
    print(f"Theta (°) minimo:\t{(360 * xmin) / (2 * np.pi)} +- {(360 * xmin_err) / (2 * np.pi)}")

    # print("----------------------------------------------- M3 -----------------------------------------------")
    # m3 = trick_quad_interp_V2(arrthetastar_rad, arrs, serrors)
    # print(m3.migrad())
    # print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")
    # xmin = m3.values["xv"]
    # xmin_err = m3.errors["xv"]
    # print(f"Theta (Rad) minimo:\t{xmin} +- {xmin_err}")
    # print(f"Theta (°) minimo:\t{(360 * xmin) / (2 * np.pi)} +- {(360 * xmin_err) / (2 * np.pi)}")

    plt.errorbar(arrthetastar_rad, arrs, serrors, linestyle = "", c = "#0e0e0e", marker = "o")

    plt.vlines(xmin, -.1, .8, linestyles = "dotted", label = f"Theta max (°) = {(360 * xmin) / (2 * np.pi):.1f} ± {(360 * xmin_err) / (2 * np.pi):.1f}")

    plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")

    lnsp = np.linspace(arrthetastar_rad[0], arrthetastar_rad[-1], 10_000)
    # plt.plot(lnsp, quad_model(lnsp, *m1.values), label = "Normale")
    plt.plot(lnsp, trick_quad_model(lnsp, *m2.values), label = "", c = "#049304")
    # plt.plot(lnsp, trick_quad_model_V2(lnsp, *m3.values), label = "Trick V2")
    plt.legend(loc = "upper left")
    plt.show()


if __name__ == "__main__":
    main()
