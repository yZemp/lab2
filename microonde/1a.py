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

arralpha = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95] # FILL
arralpha_rad = [(t * 2 * np.pi) / 360 for t in arralpha]
arrcosalpha = [abs(np.cos((t * 2 * np.pi) / 360)) for t in arralpha]
arrcosalpha2 = [np.power(t, 2) for t in arrcosalpha]

arrs = [2.49,2.48,2.44,2.39,2.33,2.19,2.14,1.93,1.87,1.58,1.5,1.09,1.05,0.68,0.57,0.24,0.15,0,0.02,0.16] # FILL
serrors = np.ones_like(arrs) * 0.09 # TODO


#####################################################################
# Functions

def model_1(x, A, B, phi):
    return  A * np.cos(x + phi) + B


def model_2(x, A, B, phi):
    return  A * np.power(np.cos(x + phi), 2) + B


def model_3(x, A, B, C, phi):
    return  A * np.power(np.cos(x + phi), 2) + B * np.cos(x) + C

def scatter(x, y, yerr):
    plt.errorbar(x, y, yerr, linestyle = "", marker = "o", c = "#018040")
    plt.show()


#####################################################################
# Interpolation
    
def interp_1(x, y, yerr, func = model_1):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1, -.5)
    m.migrad()
    m.hesse()
    return m

def interp_2(x, y, yerr, func = model_2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1, 2)
    # m.limits["phi"] = (0, 100)
    m.migrad()
    m.hesse()
    return m

def interp_3(x, y, yerr, func = model_3):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1, 1, 2)
    # m.limits["phi"] = (0, 100)
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
    m1 = interp_1(arralpha_rad, arrs, serrors)
    # print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    
    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp_2(arralpha_rad, arrs, serrors)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    

    print("----------------------------------------------- M3 -----------------------------------------------")
    m3 = interp_3(arralpha_rad, arrs, serrors)
    print(m3.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")

    plt.axes(xlabel = "Theta [rad]", ylabel = "Segnale [V]")

    plt.errorbar(arralpha_rad, arrs, serrors, linestyle = "", marker = "o", c = "#191934")
    plt.vlines(np.pi / 2, -1, 3, label = "π/2", linestyle = "dotted")

    lnsp = np.linspace(arralpha_rad[0], arralpha_rad[-1], 10_000)
    plt.plot(lnsp, model_1(lnsp, *m1.values), label = "$A\cos(x + \phi) + B$")
    plt.plot(lnsp, model_2(lnsp, *m2.values), label = "$A\cos^2(x + \phi)$ + B", c = "#a51525")
    plt.plot(lnsp, model_3(lnsp, *m3.values), label = "$A\cos^2(x + \phi) + B\cos(x + \phi) + C$", c = "#35d525")


    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
