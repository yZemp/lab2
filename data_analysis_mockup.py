import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

# import sys
# sys.path.append("/home/yzemp/Documents/Programming/lab2")

###########################################################
# vars

sheet_id = "1bjtqJHRvWQMS7QxDNb8iBOs0CKm75ioh5B47gZAaU2Y"
sheet_name = "spettro_hg_prisma"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

###########################################################
# models

def model(x):
    return x


###########################################################
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost)
    m.migrad()
    m.hesse()
    return m




#####################################################################
# Runtime

def main():

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(x, y, yerr)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    plt.errorbar(x, y, yerr, label = "Label", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(x[0] - 10, x[-1] + 10, 10_000)
    plt.plot(lnsp, model(lnsp, *m1.values), label = "Label model", c = "#a515d5")

    plt.xlabel("x Label")
    plt.ylabel("y Label")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"param = {m1.values[0]:.3f} $\pm$ {m1.errors[0]:.3f}")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
