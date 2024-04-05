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

indexes = [1, 2, 3]
lambdas = [2.7, 2.88, 2.92]
errors = [1.1, .12, .05]

#####################################################################
# Runtime

def main():
    plt.axes(xlabel = "", ylabel = "Lambda")

    plt.errorbar(indexes, lambdas, errors, linestyle = "", marker = "o", c = "#7505a5", label = "Lambda misurati")

    # plt.vlines(np.pi / 2, -1, 3, label = "π/2", linestyle = "dotted")
    plt.hlines(2.85, 0, 4, label = f"Lambda vero", linestyle = "dotted", color  = "#050505")

    # plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")

    # print(max1[1] * 180 /np.pi, max2[1] * 180 / np.pi)

    plt.legend()
    plt.show()






if __name__ == "__main__":
    main()