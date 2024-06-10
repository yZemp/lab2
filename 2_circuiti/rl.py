import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import chi2, norm, cauchy
import pandas as pd
from collections import defaultdict

import sys
sys.path.append("/home/yzemp/Documents/Programming/lab2")
from my_stats import stat, sturges


###########################################################
# vars

sheet_id = "1DR5TWcdKj22btlrAdPJKfSGy_9bbEQaM7yqhpW0MGiA"
sheet_name = "rl"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)
# print(data)

def clear_arr(arr):
    return arr[~np.isnan(arr)]


###########################################################
# models

def model(t, V0, tau):
    return V0 * (1 - np.exp(- t / tau))

def linear(x, A, B):
    return A * x + B

def quadratic(x, A, B, C):
    return A * np.power(x, 2) + B * x + C

def model_gauss(bin_edges, N, mu, sigma):
    return N * norm.cdf(bin_edges, mu, sigma)

def model_gauss_2(bin_edges, mu, sigma):
    return norm.cdf(bin_edges, mu, sigma)

def model_cauchy(bin_edges, mu, sigma):
    return cauchy.cdf(bin_edges, mu, sigma)


###########################################################
# interpolations

def interp(x, y, yerr, func = model):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m

def interp_quadratic(x, y, yerr, func = quadratic):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, .001, -10, 1)
    m.migrad()
    m.hesse()
    return m

def interp_linear(x, y, yerr, func = linear):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, -.002, .05)
    m.migrad()
    m.hesse()
    return m

#####################################################################
# Main

def main():

    # plt.errorbar(xdata, ydata, yerr, xerr, label = "Data", linestyle = "", marker = "o", markersize = 3, c = "#55d9a5", alpha = .4)

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(xdata, ydata, yerr)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    # lnsp = np.linspace(xdata[0], xdata[-1], 10_000)
    # plt.plot(lnsp, model(lnsp, *m1.values), label = "$V(t) = V_0 (1 - e^{-t/\\tau})$", c = "#a515d5")

    # plt.xlabel("Tempo [s]")
    # plt.ylabel("Tensione [V]")

    # tau = m1.values[1] * 100_000
    # tauerr = m1.errors[1] * 100_000

    # plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m1.fval / m1.ndof):.3f}, P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    # plt.plot([], [], ' ', label = f"$\\tau$ = ({tau:.3f} $\pm$ {tauerr:.3f})x10^{5} s")

    # plt.legend()
    # plt.show()

    for i in range(len(xdata)):
        yl = model(xdata[i] - xerr[i], *m1.values)
        yr = model(xdata[i] + xerr[i], *m1.values)
        diff = abs(yr - yl)
        yerr[i] = np.sqrt(np.power(yerr[i], 2) + np.power(diff, 2))

    
    plt.errorbar(xdata, ydata, yerr, label = "Data", linestyle = "", marker = "o", markersize = 3, c = "#a515d5", alpha = .9)

    print("----------------------------------------------- M2 -----------------------------------------------")
    m2 = interp(xdata, ydata, yerr)
    print(m2.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m2.fval, df = m2.ndof)}")
    
    lnsp = np.linspace(xdata[0] - .000005, xdata[-1] + .00001, 10_000)
    plt.plot(lnsp, model(lnsp, *m2.values), label = "$V(t) = V_0 (1 - e^{-t/\\tau})$", c = "#070707")

    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensione [V]")

    tau = m2.values[1] * 100_000
    tauerr = m2.errors[1] * 100_000

    plt.plot([], [], ' ', label = f"$\\chi^2_v$: {(m2.fval / m2.ndof):.3f}, P-value: {1. - chi2.cdf(m2.fval, df = m2.ndof):.4f}")
    plt.plot([], [], ' ', label = f"$\\tau$ = ({tau:.3f} $\pm$ {tauerr:.3f})x10^{5} s")

    plt.legend()
    plt.show()


    print("-------------------------------------------- ERRORS --------------------------------------------")

    # Plotting yerrs
    plt.errorbar(ydata, yerr, 0, marker = "x", label = "Samples related to their errors", markersize = 10, linestyle = "None")
    # me = interp_linear(ydata, yerr, 0)
    # print(me.migrad())

    # lnsp = np.linspace(0, 8, 10_000)
    # plt.plot(lnsp, linear(lnsp, *me.values), label = "Parabola", c = "#070707")
    # plt.plot(lnsp, quadratic(lnsp, .001, -10, 1), label = "test", c = "#070707")

    plt.xlabel("Tensione [V]")
    plt.ylabel("Errore tensione [V]")

    plt.legend()
    plt.show()


###########################################################
# Error analysis and data manipulation


# Raw data
ystep = .08

cut_start = 500
cut_end = 2100
x = clear_arr(data["x"].to_numpy())[cut_start:cut_end]
y = clear_arr(data["y"].to_numpy())[cut_start:cut_end]
# yerr = (np.ones_like(y) * ystep) / np.sqrt(12) # This is redefined later


# Zipping data to be reduced
data = sorted(zip(x, y), key = lambda tup: tup[1])

data_dict = defaultdict(list)
for i, j in data:
    data_dict[j].append(i)

# Histogram data (number of occurences in a strip, per strip)
ydata = [key for key, item in data_dict.items()]
weights = [len(item) for key, item in data_dict.items()]

# print("Momenta of occurences per strip:\t", stat(weights)["mu"], stat(weights)["stdDev"])
print(len(ydata))

plt.subplot(1, 2, 1)

plt.hist(ydata, bins = len(ydata), weights = weights, rwidth = .9, orientation = "horizontal")


# Finding average and error on the average per every strip
# plt.figure(1)

_data_statistics = [[key, stat(item)["mu"], stat(item)["stdDev"]] for key, item in data_dict.items()] # This is redefined better

# This is the actual reduced data
xdata = [stat(item)["mu"] for key, item in data_dict.items()]
ydata = ydata 
yerr = (np.ones_like(ydata) * ystep) / np.sqrt(12)
xerr = [stat(item)["stdDev"] / np.sqrt(len(item)) for key, item in data_dict.items()]

plt.subplot(1, 2, 2)

plt.errorbar(xdata, ydata, yerr = yerr, xerr = xerr, linestyle = "None")

# plt.subplot(1, 2, 2)
plt.figure(2)

# Creating a single array of centered data
centered_data = np.array([])
for key, item in data_dict.items():
    A = np.array(item - stat(item)["mu"])
    # print(A)
    centered_data = np.concatenate((centered_data, A))

# print(centered_data)

# Histrogram of normalized, centered distribution of occurences on a strip
bin_content, bin_edges, _ = plt.hist(centered_data, bins = sturges(centered_data), rwidth = .9, density = True)


# plt.hist(bin_edges, bin_content)

# Interpolation of a normal pdf
my_cost_func = ExtendedBinnedNLL(bin_content, bin_edges, model_gauss_2)
# my_minuit = Minuit(my_cost_func, N = len(centered_data), mu = stat(centered_data)["mu"], sigma = stat(centered_data)["stdDev"])
my_minuit = Minuit(my_cost_func, mu = stat(centered_data)["mu"], sigma = stat(centered_data)["stdDev"])
print(my_minuit.migrad())
print(my_minuit.valid)

lnsp = np.linspace(-.002, .002, 100_000)
# plt.plot(lnsp, normal(lnsp, my_minuit.values["mu"], my_minuit.values["sigma"]))
plt.plot(lnsp, norm.pdf(lnsp, *my_minuit.values), label = "Normal")
plt.plot(lnsp, cauchy.pdf(lnsp, *my_minuit.values), label = "Cauchy")
# plt.vlines([my_minuit.values["mu"], my_minuit.values["mu"] + my_minuit.values["sigma"], my_minuit.values["mu"] - my_minuit.values["sigma"]], 0, 200_000, linestyle = "dotted", color = "black")

plt.xlim(-.00002, .00002)

plt.legend()
plt.show()

# Trimming data (the exponential is going up)
# trim = 3
# xdata = xdata[:-trim]
# ydata = ydata[:-trim]
# yerr = yerr[:-trim]
# xerr = xerr[:-trim]

# Actual interpolation of data
if __name__ == "__main__":
    main()