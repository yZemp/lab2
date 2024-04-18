import numpy as np
import random
import scipy as sp
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import pandas as pd
# import sys
# sys.path.append("../..")

# import randgen
# import my_stats
# import funclib


#####################################################################
# Data

sheet_id = "1dRjk3ARX3-TDBIuWrPaqR57W10_ucC5kEYAa6Lzmtn0"
sheet_name = "Foglio1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

raggi1 = data["Diametro1 [cm]"].to_numpy() / 2
print(raggi1)

#####################################################################
# Functions

def model_1(x, A, B, phi):
    return  A * np.cos(x + phi) + B

#####################################################################
# Interpolation
    
def interp_1(x, y, yerr, func = model_1):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1, -.5)
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

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
