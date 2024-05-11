import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

import sys
sys.path.append("../")
import my_stats


##########################################################3
# Vars

sheet_id = "1dRjk3ARX3-TDBIuWrPaqR57W10_ucC5kEYAa6Lzmtn0"
sheet_name = "reticolo"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
# print(data)

##########################################################3
# Data

Theta_inc = 0.08226075666
lambd = 0.0000006328
d = .001

Ns1 = [1, 2, 3]
Ns2 = [-5, -4, -3, -2, -1]
Ns = [-5, -4, -3, -2, -1, 0, 1, 2, 3]
ThetaN1 = data["ThetaN1"].to_numpy()
ThetaN1 = ThetaN1[~np.isnan(ThetaN1)]
ThetaN2 = data["ThetaN2"].to_numpy()
ThetaN2 = ThetaN2[~np.isnan(ThetaN2)]
ThetaN2 = ThetaN2[::-1]
print(ThetaN1, ThetaN2)

Theta_true = [np.arccos(np.cos(Theta_inc) - (N * lambd) / d) for N in Ns]


#####################################################################
# Runtime

def main():
    plt.axes(xlabel = "N", ylabel = "$\\theta(N)$")

    plt.scatter(Ns, Theta_true, s = 100, alpha = 1, marker = "x", c = "#020202", label = "$\\theta$ teorici")

    plt.errorbar(Ns1, ThetaN1, .0011, linestyle = "None", alpha = 1, marker = "o", capsize = 0, c = "#6920a9", label = "Data")
    plt.errorbar(Ns2, ThetaN2, .0011, linestyle = "None", alpha = 1, marker = "o", capsize = 0, c = "#69a920", label = "Data")

    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
