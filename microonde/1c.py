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

arrd_max = [4.65,4.36,4.15,3.96,3.81] # FILL
arrd2_max = [np.power(d, 2) for d in arrd_max]
arrs_max = [68,69.4,70.8,72.2,73.7] # FILL
serrors_max = np.ones_like(arrs_max) * .09 # TODO

arrd = [4.39,4.65,4.05,4.36,3.92,4.15,4.1,3.8,3.96,3.81]
arrd2 = [np.power(d, 2) for d in arrd]
arrs = [67,68,69,69.4,70,70.8,71,72,72.2,73.7]
serrors = np.ones_like(arrs) * .09 # TODO


#####################################################################
# Functions

# Usable for both d and d^2
def model_max(x, A, B):
    return  A * (1 / x) + B

# Usable for both d and d^2
def model_all(x, A, B, eta, phi):
    return  A * np.cos(eta * x + phi) * (1 / x) + B

def scatter(x, y, yerr):
    plt.errorbar(x, y, yerr)
    plt.show()


#####################################################################
# Interpolation
    
def interp_max(x, y, yerr, func = model_max):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m

def interp_all(x, y, yerr, func = model_all):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1, 1, 1)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():
    scatter(arrd_max, arrs_max, serrors_max)
    # scatter(arrd, arrs, serrors)


    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp_max(arrd_max, arrs_max, serrors_max)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    
    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp_max(arrd2_max, arrs_max, serrors_max)
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



if __name__ == "__main__":
    main()
