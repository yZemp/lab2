import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

# import sys
# sys.path.append("/home/yzemp/Documents/Programming/lab2")

##########################################################3
# vars

sheet_id = "1TDCDWIADfjJNye4wf9-moEf_tEwMaD4bmi66gE59k80"
sheet_name = "due"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

req = 9.9
sigma_req = .2

corrente = data["Corrente (mA)"].to_numpy()[::-1] / 1000
sigma_corrente = data["Sigma corrente (mA)"].to_numpy()[::-1] / 1000
sigma_corrente = (data["Sigma corrente (mA)"].to_numpy()[::-1] / 1000) / np.sqrt(12)
# sigma_corrente = (np.ones_like(corrente) * .01) / 1000
potenziale = data["Diff. potenziale (V)"].to_numpy()[::-1]
sigma_potenziale = data["Sigma diff. potenziale (V)"].to_numpy()[::-1]
sigma_potenziale = (data["Sigma diff. potenziale (V)"].to_numpy()[::-1]) / np.sqrt(12)
# sigma_potenziale = np.ones_like(potenziale) * .001

# corrente = np.delete(data["Corrente (mA)"].to_numpy()[::-1] / 1000, 12)
# sigma_corrente = np.delete(data["Sigma corrente (mA)"].to_numpy()[::-1] / 1000, 12)
# potenziale = np.delete(data["Diff. potenziale (V)"].to_numpy()[::-1], 12)
# sigma_potenziale = np.delete(data["Sigma diff. potenziale (V)"].to_numpy()[::-1], 12)

##########################################################3
# models

def _model(i, ra):
    return (ra + req) * i 


def model(v, ra):
    return v / (ra + req)

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
    m1 = interp(potenziale, corrente, sigma_corrente)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")

    plt.errorbar(potenziale, corrente, sigma_corrente, xerr = sigma_potenziale, linestyle = "None", c = "#090909", markersize = msize, marker = "o")

    
    lnsp = np.linspace(potenziale[0] - 0.01, potenziale[-1] + 0.01, 10_000)
    plt.plot(lnsp, model(lnsp, *m1.values), label = "Legge di Ohm", c = "#a515d5")

    plt.xlabel("Potenziale [$V$]")
    plt.ylabel("Corrente [$A$]")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$R_a$ = {m1.values[0]:.3f} $\pm$ {m1.errors[0]:.3f} Ohm")

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


    plt.errorbar(potenziale, corrente, sigma_corrente, linestyle = "None", c = "#090909", marker = "o", markersize = 2)

    
    lnsp = np.linspace(potenziale[0] - 0.01, potenziale[-1] + 0.01, 10_000)
    plt.plot(lnsp, model(lnsp, *m2.values), label = "Legge di Ohm", c = "#a515d5")

    # plt.plot(lnsp, model(lnsp, 1, 5), label = "Tua madre", c = "#a515d5")


    plt.xlabel("Potenziale [$V$]")
    plt.ylabel("Corrente [$A$]")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m2.fval / m2.ndof):.3f}, P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$R_a$ = {m2.values[0]:.3f} $\pm$ {m2.errors[0]:.3f} Ohm")

    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
