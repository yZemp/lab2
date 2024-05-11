import numpy as np
import random
import scipy as sp
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import pandas as pd
import sys
sys.path.append(".")

# import randgen
# import my_stats
import funclib

# sheet_id = "1dRjk3ARX3-TDBIuWrPaqR57W10_ucC5kEYAa6Lzmtn0"
# sheet_name = "..."
# url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
# data = pd.read_csv(url)


#####################################################################
# Data

indexes = [0, 1, 2]
ds = np.array([0.006743172009077603, 0.006580938229483101, 0.006556269972868247])
ds = ds * 1000
errors = np.array([0.0001927106401691248, 0.0002648148369602948, 0.0001766945497248922])
errors = errors * 1000
print(ds, errors)

#####################################################################
# Runtime

def main():
    plt.axes(xlabel = "", ylabel = "Distanza [mm]")

    plt.errorbar(indexes, ds, errors, linestyle = "", marker = "o", c = "#7505a5", label = "Distanze ricavate")


    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.set_xlim(-1, 3)

    # plt.vlines(np.pi / 2, -1, 3, label = "π/2", linestyle = "dotted")
    # plt.hlines(2.85, 0, 4, label = "$\lambda_{best}$", linestyle = "dotted", color  = "#050505")

    # plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")

    # print(max1[1] * 180 /np.pi, max2[1] * 180 / np.pi)

    plt.legend()
    plt.show()






if __name__ == "__main__":
    main()