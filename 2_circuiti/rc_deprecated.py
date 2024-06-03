import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

# import sys
# sys.path.append("/home/yzemp/Documents/Programming/lab2")

###########################################################
# vars

sheet_id = "1DR5TWcdKj22btlrAdPJKfSGy_9bbEQaM7yqhpW0MGiA"
sheet_name = "rc"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
print(data)

def clear_arr(arr):
    return arr  [~np.isnan(arr)]


cut_start = 500
cut_end = -500
x = clear_arr(data["x"].to_numpy())[cut_start:cut_end]
y = clear_arr(data["y"].to_numpy())[cut_start:cut_end]
yerr = (np.ones_like(y) * .08) / np.sqrt(12)



###########################################################
# models

def model(t, V0, tau):
    return V0 * (np.exp(- t / tau))


###########################################################
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m




#####################################################################
# Runtime

def main():

    plt.errorbar(x, y, yerr, label = "Data", linestyle = "", marker = "o", markersize = 3, c = "#55d9a5", alpha = .4)

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(x, y, yerr)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    lnsp = np.linspace(x[0], x[-1], 10_000)
    plt.plot(lnsp, model(lnsp, *m1.values), label = "$V(t) = V_0 (1 - e^{-t/\\tau})$", c = "#a515d5")

    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensione [V]")

    tau = m1.values[1] * 100_000
    tauerr = m1.errors[1] * 100_000

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$\\tau$ = ({tau:.3f} $\pm$ {tauerr:.3f})x10^{5} s")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
