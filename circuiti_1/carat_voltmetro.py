import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

# import sys
# sys.path.append("/home/yzemp/Documents/Programming/lab2")

##########################################################
# vars

sheet_id = "1TDCDWIADfjJNye4wf9-moEf_tEwMaD4bmi66gE59k80"
sheet_name = "uno"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

r = 6.57
sigma_r = .06

potenziale = data["Diff. potenziale (V)"].to_numpy()
sigma_potenziale = data["Sigma diff. potenziale (V)"].to_numpy()
# sigma_potenziale = (data["Sigma diff. potenziale (V)"].to_numpy()) / np.sqrt(12)
corrente = data["Corrente (muA)"].to_numpy()
sigma_corrente = data["Sigma corrente (muA)"].to_numpy()
# sigma_corrente = (data["Sigma corrente (muA)"].to_numpy()) / np.sqrt(12)

##########################################################
# models

def model(v, rv):
    return ((1 / rv) + (1 / r)) * v 


##########################################################
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
    m1 = interp(potenziale, corrente, sigma_corrente)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

    plt.errorbar(potenziale, corrente, sigma_corrente, xerr = sigma_potenziale, linestyle = "None", c = "#090909", markersize = msize, marker = "o")

    
    lnsp = np.linspace(potenziale[0] - 0.5, potenziale[-1] + 0.5, 10_000)
    # plt.plot(lnsp, model(lnsp, *m2.values), label = "Legge di stokeazzo THE RETURN", c = "#a515d5")
    plt.plot(lnsp, model(lnsp, *m1.values), label = "Legge di Ohm", c = "#a515d5")

    plt.xlabel("Potenziale [$V$]")
    plt.ylabel("Corrente [$\\mu A$]")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$R_v$ = {m1.values[0]:.2f} $\pm$ {m1.errors[0]:.2f} MOhm")

    plt.legend()
    plt.show()



    for i in range(len(potenziale)):
        yl = model(potenziale[i] - sigma_potenziale[i], *m1.values)
        yr = model(potenziale[i] + sigma_potenziale[i], *m1.values)
        diff = abs(yr - yl)
        sigma_corrente[i] = np.sqrt(np.power(sigma_corrente[i], 2) + np.power(diff, 2))

    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp(potenziale, corrente, sigma_corrente)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")

    plt.errorbar(potenziale, corrente, sigma_corrente, linestyle = "None", c = "#090909", markersize = msize, marker = "o")

    
    lnsp = np.linspace(potenziale[0] - 0.5, potenziale[-1] + 0.5, 10_000)
    # plt.plot(lnsp, model(lnsp, *m2.values), label = "Legge di stokeazzo THE RETURN", c = "#a515d5")
    plt.plot(lnsp, model(lnsp, *m2.values), label = "Legge di Ohm", c = "#a515d5")

    plt.xlabel("Potenziale [$V$]")
    plt.ylabel("Corrente [$\\mu A$]")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m2.fval / m2.ndof):.3f}, P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$R_v$ = {m2.values[0]:.2f} $\pm$ {m2.errors[0]:.2f} MOhm")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
