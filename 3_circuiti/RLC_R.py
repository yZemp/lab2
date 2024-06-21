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
    return arr[~np.isnan(arr)]


def remove_stuff(arr):
    kill = [i for i in range(len(arr)) if not i % 2]
    kill = np.concatenate((kill, [-1, -2, -3, -4]))
    # kill = [-1, -2, -3, -4]
    return np.delete(arr, kill)

errscale = 1

# Omega (x axes)
omegas = remove_stuff(clear_arr(data["frequenza [Hz]"].to_numpy()) * 2 * np.pi)

temp_res = clear_arr(data["2Ares [V]"].to_numpy()) / 2
temp_gen = clear_arr(data["2Agen [V]"].to_numpy()) / 2
absH = remove_stuff(temp_res / temp_gen)
temp_res_err = clear_arr(data["res err [V]"].to_numpy())
temp_gen_err = clear_arr(data["gen err [V]"].to_numpy())
absHerr = remove_stuff(np.sqrt(np.power(temp_res_err / temp_gen, 2) + np.power((temp_res * temp_gen_err) / np.power(temp_gen, 2), 2)) * errscale)


phi = remove_stuff((clear_arr(data["delta_phi [°]"].to_numpy()) * 2 * np.pi) / 360)
phi_err = remove_stuff((clear_arr(data["phi err [°]"].to_numpy()) * 2 * 20 * np.pi) / 360)


###########################################################
# models

R = 10_000
# L = 15e-9

# NOTA: non considerata resistenza dell'induttore
def model_mod(omega, A, B, L, C):
    temp = omega * L - 1 / (omega * C)
    return A * np.sqrt(np.power(R, 2) / (np.power(R, 2) + np.power(temp, 2))) + B

def model_phase(omega, A, B, L, C):
    temp = omega * L - 1 / (omega * C)
    return A * (- np.arctan(temp / (R))) + B

###########################################################
# interpolations

def interp_mod(x, y, yerr, func = model_mod):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, .01, 1e-9, 1e-3) # WORKS WITH "if not i % 2" and no left side trim
    m = Minuit(my_cost, 1, .01, 1e-9, 10e-3) # WORKS WITH "not" AND WITHOUT and no left side trim
    m = Minuit(my_cost, 1, .1, 1e-9, 10e-3) # WORKS WITH BASICALLY EVERYTHING
    m = Minuit(my_cost, 1, .1, 10e-9, 10e-3)
    m.limits["L"] = (0, + np.inf)
    m.limits["C"] = (0, + np.inf)
    m.migrad()
    m.hesse()
    return m

def interp_phase(x, y, yerr, func = model_phase):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, .1, 11e-9, 1e-3)
    m.limits["L"] = (0, + np.inf)
    m.limits["C"] = (0, + np.inf)
    m.migrad()
    m.hesse()
    return m


#####################################################################
# Runtime

def main():

    print("------------------------------------------- mod -------------------------------------------")
    m1 = interp_mod(omegas, absH, absHerr)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    plt.errorbar(omegas, absH, absHerr, label = "Data", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(omegas[0] - 1_000, omegas[-1] * 2, 10_000)
    plt.plot(lnsp, model_mod(lnsp, *m1.values), label = "$|H|$", c = "#a515d5")

    plt.xlabel("Omega [Rad / s]")
    plt.ylabel("$|H|$", rotation = "horizontal")

    plt.xscale("log")
    # plt.yscale("log")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"A = {m1.values[0]:.3f} $\pm$ {m1.errors[0]:.3f}")
    # plt.plot([], [], ' ', label = f"B = {m1.values[1]:.3f} $\pm$ {m1.errors[1]:.3f}")
    plt.plot([], [], ' ', label = f"$L = (${m1.values[2] * 1e3:.1f} $\pm$ {m1.errors[2] * 1e3:.1f}$)x10^{-3}$")
    plt.plot([], [], ' ', label = f"$C = (${m1.values[3] * 1e9:.1f} $\pm$ {m1.errors[3] * 1e9:.1f}$)x10^{-9}$")

    plt.legend()
    plt.show()


    print("------------------------------------------- phase -------------------------------------------")
    m3 = interp_phase(omegas, phi, phi_err)
    print(m3.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")
    
    plt.errorbar(omegas, phi, phi_err, label = "Data", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(omegas[0] - 1_000, omegas[-1] * 2, 10_000)
    plt.plot(lnsp, model_phase(lnsp, *m3.values), label = "$\\angle H$", c = "#a515d5")

    plt.xlabel("Omega [Rad / s]")
    plt.ylabel("$\\angle H$", rotation = "horizontal")

    plt.xscale("log")
    # plt.yscale("log")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m3.fval / m3.ndof):.3f}, P-value: {1. - chi2.cdf(m3.fval, df = m3.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"A = {m3.values[0]:.3f} $\pm$ {m3.errors[0]:.3f}")
    # plt.plot([], [], ' ', label = f"B = {m3.values[1]:.3f} $\pm$ {m3.errors[1]:.3f}")
    plt.plot([], [], ' ', label = f"$L = (${m3.values[2] * 1e3:.1f} $\pm$ {m3.errors[2] * 1e3:.1f}$)x10^{-3}$")
    plt.plot([], [], ' ', label = f"$C = (${m3.values[3] * 1e9:.1f} $\pm$ {m3.errors[3] * 1e9:.1f}$)x10^{-9}$")

    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
