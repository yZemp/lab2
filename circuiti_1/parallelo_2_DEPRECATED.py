import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

# import sys
# sys.path.append("/home/yzemp/Documents/Programming/lab2")

##########################################################3
# vars


def clear_arr(arr):
    return arr[~np.isnan(arr)]

# R_v = 191.007 #TODO: is this correct?

sheet_id = "1TDCDWIADfjJNye4wf9-moEf_tEwMaD4bmi66gE59k80"
sheet_name = "tre_(ohm_parallelo)"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

corrente = data["Corrente (mA)"].to_numpy()[::-1] / 1000
corrente = clear_arr(corrente)
sigma_corrente = data["Sigma corrente (mA)"].to_numpy()[::-1] / 1000 
sigma_corrente = clear_arr(sigma_corrente)
potenziale = data["Diff. potenziale (V)"].to_numpy()[::-1]
potenziale = clear_arr(potenziale)
sigma_potenziale = data["Sigma diff. potenziale (V)"].to_numpy()[::-1]
sigma_potenziale = clear_arr(sigma_potenziale)

# print(corrente, sigma_corrente, potenziale, sigma_potenziale)
print(len(corrente))

##########################################################3
# models

def _model(V, Req):
    return V * (1 / Req)


def model(I, Req):
    return I * Req


##########################################################3
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1)
    m.migrad()
    m.hesse()
    return m



#####################################################################
# Runtime

def main():
    msize = 3

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(corrente, potenziale, sigma_potenziale)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

    plt.errorbar(corrente, potenziale, sigma_potenziale, xerr = sigma_corrente, linestyle = "None", c = "#090909", markersize = msize, marker = "o")

    
    lnsp = np.linspace(corrente[0] - 0.01, corrente[-1] + 0.01, 10_000)
    plt.plot(lnsp, model(lnsp, *m1.values), label = "Legge di Ohm", c = "#a515d5")

    plt.xlabel("Corrente [$A$]")
    plt.ylabel("Potenziale [$V$]")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$R_a$ = {m1.values[0]:.3f} $\pm$ {m1.errors[0]:.3f} Ohm")

    plt.legend()
    plt.show()



    for i in range(len(corrente)):
        yl = model(corrente[i] - sigma_corrente[i], *m1.values)
        yr = model(corrente[i] + sigma_corrente[i], *m1.values)
        diff = abs(yr - yl)
        sigma_potenziale[i] = np.sqrt(np.power(sigma_potenziale[i], 2) + np.power(diff, 2))

    
    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp(corrente, potenziale, sigma_potenziale)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")


    plt.errorbar(corrente, potenziale, sigma_potenziale, linestyle = "None", c = "#090909", marker = "o", markersize = 2)

    
    lnsp = np.linspace(corrente[0] - 0.01, corrente[-1] + 0.01, 10_000)
    plt.plot(lnsp, model(lnsp, *m2.values), label = "Legge di Ohm", c = "#a515d5")

    # plt.plot(lnsp, model(lnsp, 1, 5), label = "Tua madre", c = "#a515d5")


    plt.xlabel("Corrente [$A$]")
    plt.ylabel("Potenziale [$V$]")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m2.fval / m2.ndof):.3f}, P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$R_a$ = {m2.values[0]:.3f} $\pm$ {m2.errors[0]:.3f} Ohm")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
