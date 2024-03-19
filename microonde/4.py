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

arrtheta = [70,65,60,55,50,45,40,35,30,25,20,15] # FILL
arrtheta_rad = [(t * 2 * np.pi) / 360 for t in arrtheta]

arri = [20,25,30,35,40,45,50,55,60,65,70,75]
arri_rad = [(i * 2 * np.pi) / 360 for i in arri]

arrs = [0.07,0,0.19,0.39,0.43,0.3,0.06,0.06,0.43,0.62,0.23,0.35] # FILL
serrors = np.ones_like(arrs) * 0.09 # TODO


#####################################################################
# Functions

def model_1(x, A, B, omega, phi):
    return  A * np.cos(omega * x + phi) * (1 / x) + B


#####################################################################
# Interpolation
    
def interp_1(x, y, yerr, func = model_1):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 10, 1, 10, np.pi / 2)
    m.migrad()
    m.hesse()
    return m

#####################################################################
# Runtime

def main():

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp_1(arrtheta_rad, arrs, serrors)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    
    plt.errorbar(arrtheta_rad, arrs, serrors, linestyle = "", c = "#050560", marker = "o")
    # plt.errorbar(arri_rad, arrs, serrors, linestyle = "", c = "#05f2a1", marker = "o")
    
    lnsp = np.linspace(arrtheta_rad[0], arrtheta_rad[-1], 10_000)
    plt.plot(lnsp, model_1(lnsp, * m1.values), label = "1 / d * cos")
    plt.legend()
    plt.show()





if __name__ == "__main__":
    main()
