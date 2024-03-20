import numpy as np
import random
import scipy as sp
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import sys
sys.path.append(".")

# import randgen
# import my_stats
import funclib


#####################################################################
# Data

arrtheta = [20,25,30,35,40,45,50,55,60,65,70,75] # FILL
# arrtheta = [0,10,20,30,40,50,60,70,80] # FILL
arrthetarad = [(t * 2 * np.pi) / 360 for t in arrtheta]
# arrthetarad_sorted = list(reversed(arrthetarad))


arrs = [0.07,0,0.19,0.39,0.43,0.3,0.06,0.06,0.43,0.62,0.23,0.35] # BRAGG
serrors = np.ones_like(arrs) * .09

#####################################################################
# Functions

def model(x, A, B, omega, phi):
    return  A * np.cos(omega * x + phi) + B


#####################################################################
# Interpolation

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1, 5, 1)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():
    # scatter(arralpha_rad, arrs, serrors)
    # scatter(arrcosalpha, arrs, serrors)
    # scatter(arrcosalpha2, arrs, serrors)

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(arrthetarad, arrs, serrors)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

    plt.axes(xlabel = "Theta [rad]", ylabel = "Segnale [V]")

    plt.errorbar(arrthetarad, arrs, serrors, linestyle = "", marker = "o", c = "#050505")

    lnsp = np.linspace(arrthetarad[0], arrthetarad[-1], 10_000)
    plt.plot(lnsp, model(lnsp, *m1.values), label = "Cos", c = "#e52575")

    def fitted(x):
        return model(x, *m1.values)
    
    max1 = funclib.find_max_goldenratio(fitted, .5, .8)
    max2 = funclib.find_max_goldenratio(fitted, 1.1, 1.3)

    # plt.vlines(np.pi / 2, -1, 3, label = "π/2", linestyle = "dotted")
    plt.vlines(max1, -.3, 1, label = f"Max 1 (°) = {max1 * 180 / np.pi:.1f}", linestyle = "dotted", color  = "#05e5a5")
    plt.vlines(max2, -.3, 1, label = f"Max 2 (°) = {max2 * 180 / np.pi:.1f}", linestyle = "dotted", color  = "#05a5e5")

    plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")

    # print(max1[1] * 180 /np.pi, max2[1] * 180 / np.pi)

    plt.legend()
    plt.show()






if __name__ == "__main__":
    main()