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
sheet_name = "RL_sorted"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)


def clear_arr(arr):
    return arr[~np.isnan(arr)][2:-1]


def remove_stuff(arr):
    kill = [0, 1]
    kill = []
    return np.delete(arr, kill)

# Omega (x axes)
omegas = remove_stuff(clear_arr(data["frequenza [Hz]"].to_numpy()) * 2 * np.pi)

# First transfer function (res / gen) mod
temp_res = clear_arr(data["2Ares [V]"].to_numpy()) / 2
temp_gen = clear_arr(data["2Agen [V]"].to_numpy()) / 2
absH1 = remove_stuff(temp_res / temp_gen)
temp_res_err = clear_arr(data["res err [V]"].to_numpy())
temp_gen_err = clear_arr(data["gen err [V]"].to_numpy())
absH1err = remove_stuff(np.sqrt(np.power(temp_res_err / temp_gen, 2) + np.power((temp_res * temp_gen_err) / np.power(temp_gen, 2), 2)))

# Second transfer function (ind / gen) mod
temp_ind = clear_arr(data["2Aind [V]"].to_numpy()) / 2
temp_gen = clear_arr(data["2Agen [V]"].to_numpy()) / 2
absH2 = remove_stuff(temp_ind / temp_gen)
temp_ind_err = clear_arr(data["ind err [V]"].to_numpy())
temp_gen_err = clear_arr(data["gen err [V]"].to_numpy())
absH2err = remove_stuff(np.sqrt(np.power(temp_ind_err / temp_gen, 2) + np.power((temp_ind * temp_gen_err) / np.power(temp_gen, 2), 2)))

errscale = 12

# First transfer function (res / gen) phase
phi = remove_stuff((clear_arr(data["delta_phi [°]"].to_numpy()) * 2 * np.pi) / 360)
phi_err = remove_stuff((clear_arr(data["phi err [°]"].to_numpy()) * 2 * errscale * np.pi) / 360)

# First transfer function (res / gen) phase
chi = remove_stuff((clear_arr(data["delta_chi [°]"].to_numpy()) * 2 * np.pi) / 360)
chi_err = remove_stuff((clear_arr(data["chi err [°]"].to_numpy()) * 2 * errscale * np.pi) / 360)


###########################################################
# models

R = 10_000 # TBD
# L = 1e-3

def model_mod_H1(omega, A, B, L, Rl):
    temp = omega * L
    return A * np.sqrt(np.power(Rl, 2) + np.power(temp, 2)) / np.sqrt(np.power(R + Rl, 2) + np.power(temp, 2)) + B
    
def model_mod_H2(omega, A, B, L, Rl):
    temp = omega * L
    return A * np.sqrt(np.power(R, 2) / np.sqrt(np.power(R + Rl, 2) + np.power(temp, 2))) + B
    

def model_phase_H1(omega, A, B, L, Rl):
    temp = omega * L
    return A * (np.arctan(temp / Rl) - np.arctan(temp / (R + Rl))) + B

def model_phase_H2(omega, A, B, L, Rl):
    temp = omega * L
    return A * ( - np.arctan(temp / (R + Rl))) + B


###########################################################
# interpolations

def interp_mod_H1(x, y, yerr, func = model_mod_H1):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, .1, 1e-3, 60)
    m.limits["A"] = (-1, 1)
    m.limits["L"] = (0, + np.inf)
    m.limits["Rl"] = (50, 70)
    m.migrad()
    m.hesse()
    return m

def interp_mod_H2(x, y, yerr, func = model_mod_H2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, .1, 1e-3, 60)
    m.limits["L"] = (0, + np.inf)
    m.limits["Rl"] = (50, 70)
    m.migrad()
    m.hesse()
    return m


def interp_phase_H1(x, y, yerr, func = model_phase_H1):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, .1, 1e-3, 60)
    m.limits["L"] = (0, + np.inf)
    m.limits["Rl"] = (50, 70)
    m.migrad()
    m.hesse()
    return m



def interp_phase_H2(x, y, yerr, func = model_phase_H2):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, -1, .1, 1e-3, 50)
    m.limits["L"] = (0, + np.inf)
    m.limits["Rl"] = (0, 70)
    m.migrad()
    m.hesse()
    return m

#####################################################################
# Runtime

def main():

    print("------------------------------------------- H1 mod -------------------------------------------")
    m1 = interp_mod_H1(omegas, absH1, absH1err)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    plt.errorbar(omegas, absH1, absH1err, label = "Label", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(omegas[0] - 1_000, omegas[-1] * 2, 10_000)
    plt.plot(lnsp, model_mod_H1(lnsp, *m1.values), label = "Label model", c = "#a515d5")

    plt.xlabel("Omega [Rad / s]")
    plt.ylabel("$|H|$", rotation = "horizontal")

    plt.xscale("log")
    # plt.yscale("log")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"A = {m1.values[0]:.3f} $\pm$ {m1.errors[0]:.3f}")
    # plt.plot([], [], ' ', label = f"B = {m1.values[1]:.3f} $\pm$ {m1.errors[1]:.3f}")
    plt.plot([], [], ' ', label = f"$L = (${m1.values[2] * 1e3:.1f} $\pm$ {m1.errors[2] * 1e3:.1f}$)x10^{-3}$")

    plt.legend()
    plt.show()


    print("------------------------------------------- H2 mod -------------------------------------------")
    m2 = interp_mod_H2(omegas, absH2, absH2err)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    
    plt.errorbar(omegas, absH2, absH2err, label = "Label", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(omegas[0] - 1_000, omegas[-1] * 2, 10_000)
    plt.plot(lnsp, model_mod_H2(lnsp, *m2.values), label = "Label model", c = "#a515d5")

    plt.xlabel("Omega [Rad / s]")
    plt.ylabel("$|H|$", rotation = "horizontal")

    plt.xscale("log")
    # plt.yscale("log")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m2.fval / m2.ndof):.3f}, P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"A = {m2.values[0]:.3f} $\pm$ {m2.errors[0]:.3f}")
    # plt.plot([], [], ' ', label = f"B = {m2.values[1]:.3f} $\pm$ {m2.errors[1]:.3f}")
    plt.plot([], [], ' ', label = f"$L = (${m2.values[2] * 1e3:.1f} $\pm$ {m2.errors[2] * 1e3:.1f}$)x10^{-3}$")

    plt.legend()
    plt.show()



    print("------------------------------------------- H1 phase -------------------------------------------")
    m3 = interp_phase_H1(omegas, phi, phi_err)
    print(m3.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m3.fval, df = m3.ndof)}")
    
    plt.errorbar(omegas, phi, phi_err, label = "Label", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(omegas[0] - 1_000, omegas[-1] * 2, 10_000)
    plt.plot(lnsp, model_phase_H1(lnsp, *m3.values), label = "Label model", c = "#a515d5")

    plt.xlabel("Omega [Rad / s]")
    plt.ylabel("$\\angle H$", rotation = "horizontal")

    plt.xscale("log")
    # plt.yscale("log")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m3.fval / m3.ndof):.3f}, P-value: {1. - chi2.cdf(m3.fval, df = m3.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"A = {m3.values[0]:.3f} $\pm$ {m3.errors[0]:.3f}")
    # plt.plot([], [], ' ', label = f"B = {m3.values[1]:.3f} $\pm$ {m3.errors[1]:.3f}")
    plt.plot([], [], ' ', label = f"$L = (${m3.values[2] * 1e3:.1f} $\pm$ {m3.errors[2] * 1e3:.1f}$)x10^{-3}$")

    plt.legend()
    plt.show()


    print("------------------------------------------- H2 phase -------------------------------------------")
    m4 = interp_phase_H2(omegas, chi, chi_err)
    print(m4.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m4.fval, df = m4.ndof)}")
    
    plt.errorbar(omegas, chi, chi_err, label = "Label", linestyle = "", marker = "o", c = "#151515")
    lnsp = np.linspace(omegas[0] - 1_000, omegas[-1] * 2, 10_000)
    plt.plot(lnsp, model_phase_H2(lnsp, *m4.values), label = "Label model", c = "#a515d5")

    plt.xlabel("Omega [Rad / s]")
    plt.ylabel("$\\angle H$", rotation = "horizontal")

    plt.xscale("log")
    # plt.yscale("log")

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m4.fval / m4.ndof):.3f}, P-value: {1. - chi2.cdf(m4.fval, df = m4.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"A = {m4.values[0]:.3f} $\pm$ {m4.errors[0]:.3f}")
    # plt.plot([], [], ' ', label = f"B = {m4.values[1]:.3f} $\pm$ {m4.errors[1]:.3f}")
    plt.plot([], [], ' ', label = f"$L = (${m4.values[2] * 1e3:.1f} $\pm$ {m4.errors[2] * 1e3:.1f}$)x10^{-3}$")

    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
