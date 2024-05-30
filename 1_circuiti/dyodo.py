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
sheet_name = "dyodo"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

# kdiodo = 38.6

cut0 = 0
# cut0 = 4

corrente0 = data["Corrente (muA)"].to_numpy()[:-7 or None][cut0:]
sigma_corrente0 = data["Sigma corrente (muA)"].to_numpy()[:-7 or None][cut0:]
potenziale0 = data["Diff. potenziale (V) 0"].to_numpy()[:-7 or None][cut0:]
sigma_potenziale0 = data["Sigma diff. potenziale (V) 0"].to_numpy()[:-7 or None][cut0:]

cut1 = 0
# cut1 = 3

corrente1 = data["Corrente (mA) 1"].to_numpy()[:-cut1 or None]
sigma_corrente1 = data["Sigma corrente (mA) 1"].to_numpy()[:-cut1 or None]
potenziale1 = data["Diff. potenziale (V) 1"].to_numpy()[:-cut1 or None]
sigma_potenziale1 = data["Sigma diff. potenziale (V) 1"].to_numpy()[:-cut1 or None]

cut2 = 0
# cut2 = 8

corrente2 = data["Corrente (mA) 2"].to_numpy()[:-cut2 or None]
sigma_corrente2 = data["Sigma corrente (mA) 2"].to_numpy()[:-cut2 or None]
potenziale2 = data["Diff. potenziale (V) 2"].to_numpy()[:-cut2 or None]
sigma_potenziale2 = data["Sigma diff. potenziale (V) 2"].to_numpy()[:-cut2 or None]

# corrente2 = data["Corrente (mA) 2"].to_numpy()
# sigma_corrente2 = data["Sigma corrente (mA) 2"].to_numpy()
# potenziale2 = data["Diff. potenziale (V) 2"].to_numpy()
# sigma_potenziale2 = data["Sigma diff. potenziale (V) 2"].to_numpy()


corrente = np.concatenate((corrente0 / 1000, corrente1, corrente2))
# corrente = np.concatenate((corrente1, corrente2))
corrente = corrente[~np.isnan(corrente)]
sigma_corrente = np.concatenate((sigma_corrente0 / 1000, sigma_corrente1, sigma_corrente2))
# sigma_corrente = np.concatenate((sigma_corrente1, sigma_corrente2))
sigma_corrente = sigma_corrente[~np.isnan(sigma_corrente)]
# sigma_corrente = sigma_corrente[~np.isnan(sigma_corrente)] / np.sqrt(12)
potenziale = np.concatenate((potenziale0, potenziale1, potenziale2))
# potenziale = np.concatenate((potenziale1, potenziale2))
potenziale = potenziale[~np.isnan(potenziale)]
sigma_potenziale = np.concatenate((sigma_potenziale0, sigma_potenziale1, sigma_potenziale2))
# sigma_potenziale = np.concatenate((sigma_potenziale1, sigma_potenziale2))
sigma_potenziale = sigma_potenziale[~np.isnan(sigma_potenziale)]
# sigma_potenziale = sigma_potenziale[~np.isnan(sigma_potenziale)] / np.sqrt(12)

print(corrente, "\n", sigma_corrente, "\n", potenziale, "\n", sigma_potenziale, "\n")


# plt.errorbar(potenziale0, corrente0 / 1000, yerr = sigma_corrente0 / 1000, xerr = sigma_potenziale0)
# plt.errorbar(potenziale1, corrente1, yerr = sigma_corrente1, xerr = sigma_potenziale1)
# plt.errorbar(potenziale2, corrente2, yerr = sigma_corrente2, xerr = sigma_potenziale2)
# plt.show()

##########################################################3
# models

def model(V, kdiodo, I_0, g):
    return I_0 * (np.exp((kdiodo * V) / g) - 1)

# def model_joke(V, I_0, g, A, omega, phi):
#     return I_0 * (np.exp((kdiodo * V) / g) - 1) + A * np.sin(omega * g + phi)


##########################################################3
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 38.6, .001, 10)
    m.limits["kdiodo"] = (35, 45)
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

    # plt.errorbar(potenziale, corrente, yerr = sigma_corrente, xerr = sigma_potenziale, linestyle = "None", c = "#090909")

    # plt.yscale("log")
    
    # lnsp = np.linspace(potenziale[0] - 0.01, potenziale[-1] + 0.01, 10_000)
    # plt.plot(lnsp, model(lnsp, *m1.values), label = "Legge", c = "#a515d5")

    # plt.plot(lnsp, model(lnsp, 1, 5), label = "Tua madre", c = "#a515d5")

    # plt.xlabel("$Potenziale [V]$")
    # plt.ylabel("$Corrente [mA]$")

    # plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"$I_0$ = {m1.values[0]:.2f} $\pm$ {m1.errors[0]:.2f}")
    # plt.plot([], [], ' ', label = f"$g$ = {m1.values[1]:.2f} $\pm$ {m1.errors[1]:.2f}")

    # plt.legend()
    # plt.show()



    for i in range(len(potenziale)):
        yl = model(potenziale[i] - sigma_potenziale[i], *m1.values)
        yr = model(potenziale[i] + sigma_potenziale[i], *m1.values)
        diff = abs(yr - yl)
        sigma_corrente[i] = np.sqrt(np.power(sigma_corrente[i], 2) + np.power(diff, 2))

    
    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp(potenziale, corrente, sigma_corrente)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")

    fig, axes = plt.subplots(1, 2)

    axes[0].errorbar(potenziale, corrente, yerr = sigma_corrente, linestyle = "None", c = "#090909", marker = "o", markersize = msize)

    lnsp = np.linspace(potenziale[0] - 0.01, potenziale[-1] + 0.01, 10_000)
    axes[0].plot(lnsp, model(lnsp, *m2.values), label = "Legge di Shockley", c = "#a515d5")

    axes[0].plot([], [], ' ', label = f"$\\chi^2_v$: {(m2.fval / m2.ndof):.3f}, P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    axes[0].plot([], [], ' ', label = f"Kdiodo = ({m2.values[0]:.0f} $\pm$ {m2.errors[0]:.0f})")
    axes[0].plot([], [], ' ', label = f"g = ({m2.values[2]:.2f} $\pm$ {m2.errors[2]:.2f})")

    axes[0].legend(loc = "upper left")

    axes[1].set_yscale("log")

    axes[1].errorbar(potenziale, corrente, yerr = sigma_corrente, linestyle = "None", c = "#090909", marker = "o", markersize = msize)

    lnsp = np.linspace(potenziale[0] - 0.01, potenziale[-1] + 0.01, 10_000)
    axes[1].plot(lnsp, model(lnsp, *m2.values), label = "Legge di Shockley", c = "#a515d5")

    # plt.plot(lnsp, model(lnsp, 1, 5), label = "Tua madre", c = "#a515d5")

    for ax in axes:
        ax.set(xlabel = "Potenziale $[V]$", ylabel = "Corrente $[mA]$")

    axes[1].plot([], [], ' ', label = f"$\\chi^2_v$: {(m2.fval / m2.ndof):.3f}, P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    axes[1].plot([], [], ' ', label = f"Kdiodo = ({m2.values[0]:.0f} $\pm$ {m2.errors[0]:.0f})")
    axes[1].plot([], [], ' ', label = f"g = ({m2.values[2]:.2f} $\pm$ {m2.errors[2]:.2f})")

    axes[1].legend()

    plt.show()


if __name__ == "__main__":
    main()
