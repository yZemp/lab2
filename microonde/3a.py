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

arrd = [0.116,0.192,0.227,0.252,0.272,0.284,0.294] # FILL
# arrd = [d * 2 for d in arrd]

arrs = [2.33,2.31,2.29,2.29,2.29,2.26,2.3] # FILL
serrors = np.ones_like(arrs) * 0.09 # TODO


#####################################################################
# Functions

def model_1(x, A, B):
    return  A * (1 / x) + B


def model_2(x, A, B):
    return  A * (1 / np.power(x, 2)) + B


def model_3(x, A, B, pazzo):
    return  A * (1 / np.power(x, 2)) + pazzo * (1 / x) + B
# Questo non aveva senso o sbaglio?

#####################################################################
# Interpolation
    
def interp_1(x, y, yerr, func = model_1):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m

def interp_2(x, y, yerr, func = model_2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m

def interp_3(x, y, yerr, func = model_3):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1, 1)
    m.migrad()
    m.hesse()
    return m

#####################################################################
# Runtime

def main():

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp_1(arrd, arrs, serrors)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp_2(arrd, arrs, serrors)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    

    plt.errorbar(arrd, arrs, serrors, linestyle = "", c = "black", marker = "o")
    
    lnsp = np.linspace(arrd[0], arrd[-1], 10_000)
    plt.plot(lnsp, model_1(lnsp, * m1.values), label = "1 / d\n model_1")
    plt.plot(lnsp, model_2(lnsp, * m2.values), label = "1 / d^2\n model_2")
    plt.legend()
    plt.show()





if __name__ == "__main__":
    main()
