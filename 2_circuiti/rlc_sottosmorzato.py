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
sheet_name = "rlc_sott"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
print(data)

def clear_arr(arr):
    return arr[~np.isnan(arr)]


cut_start = 100
cut_end = -1
x = clear_arr(data["x"].to_numpy())[cut_start:cut_end]
y = clear_arr(data["y"].to_numpy())[cut_start:cut_end]
yerr = (np.ones_like(y) * .012) / np.sqrt(12)


###########################################################
# models

def model(t, V0, gamma, omega_0, phi):
    return V0 * np.exp(- gamma * t) * np.sin(omega_0 * t + phi)

def model_ext(t, V0, gamma, omega, phi):
    omega_0 = np.sqrt(np.power(omega, 2) - np.power(gamma, 2))
    return V0 * np.exp(- gamma * t) * np.sin(omega_0 * t + phi)

###########################################################
# interpolations

def interp(x, y, yerr, func = model_ext):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 600, 1300, 15000, 3)
    m.migrad()
    m.hesse()
    return m




#####################################################################
# Runtime

def main():

    plt.errorbar(x, y, yerr, label = "Data", linestyle = "", marker = "o", markersize = 5, c = "#55d9a5", alpha = .4)

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(x, y, yerr)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    lnsp = np.linspace(0, 0.003, 100_000)
    plt.plot(lnsp, model_ext(lnsp, *m1.values), label = "$V(t) = V_0 (1 - e^{-t/\\tau})$", c = "#a515d5")

    # plt.plot(lnsp, 600 * np.exp(- 1300 * lnsp) * np.sin(np.sqrt(np.power(1300, 2) - np.power(15000, 2)) * lnsp + 1), label = "test")
    # plt.plot(lnsp, model_ext(lnsp, 600, 1300, 15_000, 3), label = "test")

    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensione [V]")

    gamma = m1.values[1] * 1
    gammaerr = m1.errors[1] * 1

    omega = m1.values[2] * 1
    omegaerr = m1.errors[2] * 1

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$\\gamma$ = ({gamma:.0f} $\pm$ {gammaerr:.0f})s")
    plt.plot([], [], ' ', label = f"$\\omega$ = ({omega:.0f} $\pm$ {omegaerr:.0f})s")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
