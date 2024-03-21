import numpy as np
import random
import scipy as sp
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import sys
sys.path.append(".")

# import randgen
import my_stats
# import funclib

#####################################################################
# Finding the error
# deltas = [.8, 1.1, .8, .6, .9, .9, .6, .7, .75, .75, .75, .6, .7, .75, .65]
# stats = my_stats.stat(deltas)
# print(stats["mu"], stats["stdDev"])

#####################################################################
# Data

# arrd_max = [11.2,12,12.8,13.9,14.5,15.4,16.3,16.9,18.3,19.8,21.3,22.8,27.1,31.5,35.7,40.1,41.4,44.4,45.6,48.7,50.1,51.4] # FILL
# arrs_max = [3.82,3.18,3.38,3.04,3.02,3.01,2.88,3,2.88,2.84,2.76,2.68,2.48,2.24,2.06,1.85,1.77,1.61,1.5,1.57,1.55,1.51] # FILL
# serrors_max = np.ones_like(arrs_max) * .09 # TODO

arrd = [67,68,69,69.4,70,70.8,71,72,72.2,73.7]
arrs = [4.39,4.65,4.05,4.36,3.92,4.15,4.1,3.8,3.96,3.81]
serrors = np.ones_like(arrs) * .09

#####################################################################
# Functions

def model_cos(x, A, B, omega, phi):
    return A * abs(np.cos(omega * x + phi)) + B

def model_jack(x, A, B, C, omega, phi):
    return A * abs(np.cos(omega * x + phi)) + B * np.power(x, - 2) + C

# Usable for both d and d^2
# def model_all(x, A, B, eta, phi):
#     return  A * np.cos(eta * x + phi) * (1 / x) + B

def scatter(x, y, yerr):
    plt.errorbar(x, y, yerr)
    plt.show()


#####################################################################
# Interpolation
    

def interp_cos(x, y, yerr, func = model_cos):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 10, 2, -1)
    m.migrad()
    m.hesse()
    return m

def interp_jack(x, y, yerr, func = model_jack):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, -1, 0., 1, 2, -1)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():
    # scatter(arrd, arrs, serrors)

    print("----------------------------------------------- Mcos -----------------------------------------------")
    mcos = interp_cos(arrd, arrs, serrors)
    print(mcos.migrad())
    print(f"Pval:\t{1. - chi2.cdf(mcos.fval, df = mcos.ndof)}")

    print("----------------------------------------------- MJ -----------------------------------------------")
    mj = interp_jack(arrd, arrs, serrors)
    print(mj.migrad())
    print(f"Pval:\t{1. - chi2.cdf(mj.fval, df = mj.ndof)}")
    
    plt.axes(xlabel = "d [cm]", ylabel = "Segnale [V]")

    plt.errorbar(arrd, arrs, serrors, linestyle = "", c = "#0e0e0e", marker = "o")

    lnsp = np.linspace(arrd[0], arrd[-1], 10_000)
    # plt.plot(lnsp, model_max_1(lnsp, * m1.values), label = "1 / d\n model_1")
    # plt.plot(lnsp, model_max_2(lnsp, * m2.values), label = "1 / d^2\n model_2")

    plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(mcos.fval, df = mcos.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(mj.fval, df = mj.ndof):.4f}")

    plt.plot(lnsp, model_cos(lnsp, * mcos.values), label = "Fronte d'onda piano", color = "#7525e5")
    # plt.plot(lnsp, model_jack(lnsp, * mj.values), label = "Fronte d'onda sferico/ellittico", color = "#7525e5")
    plt.legend()
    plt.show()

    # print("----------------------------------------------- M3 -----------------------------------------------")
    # m3 = interp_all(arrd, arrs, serrors)
    # print(m3.migrad())
    # print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")

    
    # print("----------------------------------------------- M4 -----------------------------------------------")
    # m4 = interp_all(arrd2, arrs, serrors)
    # print(m4.migrad())
    # print(f"Pval:\t{1. - chi2.cdf(m4.fval, df = m4.ndof)}")



if __name__ == "__main__":
    main()
