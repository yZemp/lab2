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

arrtheta = [180,170,160,150,140,130,120,110,100] # FILL
# arrtheta = [0,10,20,30,40,50,60,70,80] # FILL
arrthetarad = [((180 - t) * 2 * np.pi) / 360 for t in arrtheta]
arrthetarad = arrthetarad[1:-1]

arrs1 = [4.3,3.43,1.93,0.71,0.22,0.07,0.03,0.02,0] # FILL # ATTACCATO
arrs1 = arrs1[1:-1]
arrs2 = [2.17,1.98,1.1,0.42,0.13,0.04,0,0,0] # FILL #STACCATO
arrs2 = arrs2[1:-1]
serrors = np.ones_like(arrs1) * 0.09 # TODO


#####################################################################
# Functions

def model_cos(x, A, B, omega, phi):
    return A * np.cos(omega * x + phi) + B

# def model_cos_corretto(x, A, B, omega, phi):
#     return  A * np.cos(omega * x + phi) + B

def model_ellisse(x, a, b):
    return  (2 * a * np.power(b, 2) * np.cos(x)) / (np.power(b, 2) * np.power(np.cos(x), 2) + np.power(a, 2) * np.power(np.sin(x), 2))

#####################################################################
# Interpolation
    
def interp_cos(x, y, yerr, func = model_cos):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, -1, -1, 1, 0.)
    m.migrad()
    m.hesse()
    return m

def interp_ellisse(x, y, yerr, func = model_ellisse):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
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
    m1 = interp_cos(arrthetarad, arrs1, serrors)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp_cos(arrthetarad, arrs2, serrors)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")

    print("----------------------------------------------- M3 -----------------------------------------------")
    m3 = interp_ellisse(arrthetarad, arrs1, serrors)
    print(m3.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")

    print("----------------------------------------------- M4 -----------------------------------------------")
    m4 = interp_ellisse(arrthetarad, arrs2, serrors)
    print(m4.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m4.fval, df = m4.ndof)}")

    plt.axes(xlabel = "Theta [rad]", ylabel = "Segnale [V]")

    plt.errorbar(arrthetarad, arrs1, serrors, linestyle = "", marker = "o", c = "#05e545")
    plt.errorbar(arrthetarad, arrs2, serrors, linestyle = "", marker = "o", c = "#e50545")
    # plt.vlines(np.pi / 2, -1, 3, label = "π/2", linestyle = "dotted")

    lnsp = np.linspace(arrthetarad[0], arrthetarad[-1], 10_000)

    # plt.plot([], [], ' ', label = f"P-value (attaccato): {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"P-value (staccato): {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")

    # plt.plot(lnsp, model_cos(lnsp, *m1.values), label = "Attaccato")
    # plt.plot(lnsp, model_cos(lnsp, *m2.values), label = "Staccato")


    plt.plot([], [], ' ', label = f"P-value (attaccato): {1. - chi2.cdf(m3.fval, df = m3.ndof):.4f}")
    plt.plot([], [], ' ', label = f"P-value (staccato): {1. - chi2.cdf(m4.fval, df = m4.ndof):.4f}")


    plt.plot(lnsp, model_ellisse(lnsp, *m3.values), label = "Attaccato")
    plt.plot(lnsp, model_ellisse(lnsp, *m4.values), label = "Staccato")

    plt.legend()
    plt.show()






if __name__ == "__main__":
    main()
