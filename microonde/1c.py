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

arrd_max = [11.2,12,12.8,13.9,14.5,15.4,16.3,16.9,18.3,19.8,21.3,22.8,27.1,31.5,35.7,40.1,41.4,44.4,45.6,48.7,50.1,51.4] # FILL
# arrd2_max = [np.power(d, 2) for d in arrd_max]
arrs_max = [3.82,3.18,3.38,3.04,3.02,3.01,2.88,3,2.88,2.84,2.76,2.68,2.48,2.24,2.06,1.85,1.77,1.61,1.5,1.57,1.55,1.51] # FILL
serrors_max = np.ones_like(arrs_max) * .09 # TODO

# arrd = [4.39,4.65,4.05,4.36,3.92,4.15,4.1,3.8,3.96,3.81]
# arrd2 = [np.power(d, 2) for d in arrd]
# arrs = [67,68,69,69.4,70,70.8,71,72,72.2,73.7]
# serrors = np.ones_like(arrs) * .09 # TODO


#####################################################################
# Functions

# Usable for both d and d^2
def model_max(x, A, B):
    return  A * (1 / x) + B

def model_max_2(x, A, B):
    return  A * (1 / np.power(x, 2)) + B

def model_pazzo(x, A, B, C, D, omega, phi):
    return  A * np.cos(omega * x + phi) + B * np.power(x, -1) + C * np.power(x, -2) + D

# Usable for both d and d^2
# def model_all(x, A, B, eta, phi):
#     return  A * np.cos(eta * x + phi) * (1 / x) + B

# def scatter(x, y, yerr):
#     plt.errorbar(x, y, yerr)
#     plt.show()


#####################################################################
# Interpolation
    
def interp_max(x, y, yerr, func = model_max):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, .1, 1)
    m.migrad()
    m.hesse()
    return m

def interp_max_2(x, y, yerr, func = model_max_2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, .1, 1)
    m.migrad()
    m.hesse()
    return m

# def interp_all(x, y, yerr, func = model_all):
#     my_cost = cost.LeastSquares(x, y, yerr, func)
#     m = Minuit(my_cost, 1, 1, 1, 1)
#     m.migrad()
#     m.hesse()
#     return m


#####################################################################
# Runtime

def main():
    # scatter(arrd_max, arrs_max, serrors_max)
    # scatter(arrd, arrs, serrors)


    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp_max(arrd_max, arrs_max, serrors_max)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    
    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp_max_2(arrd_max, arrs_max, serrors_max)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    

    # print("----------------------------------------------- M3 -----------------------------------------------")
    # m3 = interp_all(arrd, arrs, serrors)
    # print(m3.migrad())
    # print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")

    
    # print("----------------------------------------------- M4 -----------------------------------------------")
    # m4 = interp_all(arrd2, arrs, serrors)
    # print(m4.migrad())
    # print(f"Pval:\t{1. - chi2.cdf(m4.fval, df = m4.ndof)}")

    plt.axes(xlabel = "d [cm]", ylabel = "Segnale [V]")

    plt.errorbar(arrd_max, arrs_max, serrors_max, linestyle = "", marker = "o", c = "#0e0e0e")

    lnsp = np.linspace(arrd_max[0], arrd_max[-1], 10_000)
    # plt.plot(lnsp, quad_model(lnsp, *m1.values), label = "Normale")
    plt.plot(lnsp, model_max(lnsp, *m1.values), label = "d")
    plt.plot(lnsp, model_max_2(lnsp, *m2.values), label = "d^2")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
