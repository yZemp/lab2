import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

# import sys
# sys.path.append("/home/yzemp/Documents/Programming/lab2")

###########################################################
# vars

sheet_id = "13lPIFW2CK69SiacRqfcQitab0GxV_9aOefAAglwiJlE"
sheet_name = "RLC_R"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)


def clear_arr(arr):
    return arr[~np.isnan(arr)][2:-1]


# Omega (x axes)
omegas = clear_arr(data["frequenza [Hz]"].to_numpy()) * 2 * np.pi

temp_res = clear_arr(data["2Ares [V]"].to_numpy()) / 2
temp_gen = clear_arr(data["2Agen [V]"].to_numpy()) / 2
absH1 = temp_res / temp_gen
temp_res_err = clear_arr(data["res err [V]"].to_numpy())
temp_gen_err = clear_arr(data["gen err [V]"].to_numpy())
absH1err = np.sqrt(np.power(temp_res_err / temp_gen, 2) + np.power((temp_res * temp_gen_err) / np.power(temp_gen, 2), 2))

phi = (clear_arr(data["delta_phi [°]"].to_numpy()) * 2 * np.pi) / 360
phi_err = (clear_arr(data["phi err [°]"].to_numpy()) * 2 * 10 * np.pi) / 360


###########################################################
# models

R = 10_000 # TBD
# L = 15e-9

def model_mod(omega, A, B, L, Rl):
    temp = omega * L
    return A * np.sqrt(np.power(Rl, 2) + np.power(temp, 2)) / np.sqrt(np.power(R + Rl, 2) + np.power(temp, 2)) + B

def model_phase(omega, A, B, L, Rl):
    temp = omega * L
    return A * (np.arctan(temp / Rl) - np.arctan(temp / (R + Rl))) + B

###########################################################
# interpolations

def interp_mod(x, y, yerr, func = model_mod):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, .1, 11e-9, 60)
    m.limits["A"] = (-1, 1)
    m.limits["L"] = (0, + np.inf)
    m.limits["Rl"] = (50, 70)
    m.migrad()
    m.hesse()
    return m

def interp_phase(x, y, yerr, func = model_phase):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, .1, 11e-9, 60)
    m.limits["L"] = (0, + np.inf)
    m.limits["Rl"] = (50, 70)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():

    print("------------------------------------------- mod -------------------------------------------")
    m1 = interp_mod(omegas, absH1, absH1err)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    plt.errorbar(omegas, absH1, absH1err, label = "Label", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(omegas[0] - 1_000, omegas[-1] + 1_000, 10_000)
    plt.plot(lnsp, model_mod(lnsp, *m1.values), label = "Label model", c = "#a515d5")

    plt.xlabel("Omega [Rad / s]")
    plt.ylabel("H1 mod")

    plt.xscale("log")
    # plt.yscale("log")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"A = {m1.values[0]:.3f} $\pm$ {m1.errors[0]:.3f}")
    # plt.plot([], [], ' ', label = f"B = {m1.values[1]:.3f} $\pm$ {m1.errors[1]:.3f}")
    plt.plot([], [], ' ', label = f"$L = (${m1.values[2] * 1e9:.1f} $\pm$ {m1.errors[2] * 1e9:.1f}$)x10^{-9}$")

    plt.legend()
    plt.show()


    print("------------------------------------------- phase -------------------------------------------")
    m3 = interp_phase(omegas, phi, phi_err)
    print(m3.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")
    
    plt.errorbar(omegas, phi, phi_err, label = "Label", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(omegas[0] - 1_000, omegas[-1] + 1_000, 10_000)
    plt.plot(lnsp, model_phase(lnsp, *m3.values), label = "Label model", c = "#a515d5")

    plt.xlabel("Omega [Rad / s]")
    plt.ylabel("H1 phase")

    plt.xscale("log")
    # plt.yscale("log")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m3.fval / m3.ndof):.3f}, P-value: {1. - chi2.cdf(m3.fval, df = m3.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"A = {m3.values[0]:.3f} $\pm$ {m3.errors[0]:.3f}")
    # plt.plot([], [], ' ', label = f"B = {m3.values[1]:.3f} $\pm$ {m3.errors[1]:.3f}")
    plt.plot([], [], ' ', label = f"$L = (${m3.values[2] * 1e9:.1f} $\pm$ {m3.errors[2] * 1e9:.1f}$)x10^{-9}$")

    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
