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

def model_gauss(bin_edges, N, mu, sigma):
    return N * norm.cdf(bin_edges, mu, sigma)

def model_gauss_2(bin_edges, mu, sigma):
    return norm.cdf(bin_edges, mu, sigma)

def model_cauchy(bin_edges, mu, sigma):
    return cauchy.cdf(bin_edges, mu, sigma)


###########################################################
# interpolations

def interp(bin_content, bin_edges, model_distrib = model_gauss_2):
    c = ExtendedBinnedNLL(bin_content, bin_edges, model_distrib)
    # my_minuit = Minuit(my_cost_func, N = len(centered_data), mu = stat(centered_data)["mu"], sigma = stat(centered_data)["stdDev"])
    m = Minuit(c, mu = stat(centered_data)["mu"], sigma = stat(centered_data)["stdDev"])
    # print(my_minuit.migrad())
    # print(my_minuit.valid)
    m.migrad()
    m.hesse()
    return m


###########################################################
# Error analysis and data manipulation


# Raw data
ystep = .08

cut_start = 500
cut_end = 2100
x = clear_arr(data["x"].to_numpy())[cut_start:cut_end]
y = clear_arr(data["y"].to_numpy())[cut_start:cut_end]
yerr = (np.ones_like(y) * ystep) / np.sqrt(12) # This is redefined later

plt.figure(0)
plt.errorbar(x, y, yerr, linestyle = "none", color = "#55d9a5", alpha = .5, marker = "o", markersize = 3)
plt.show()


# Zipping data to be reduced
data = sorted(zip(x, y), key = lambda tup: tup[1])

data_dict = defaultdict(list)
for i, j in data:
    data_dict[j].append(i)

# Histogram data (number of occurences in a strip, per strip)
ydata = [key for key, item in data_dict.items()]
weights = [len(item) for key, item in data_dict.items()]

plt.figure(1)
plt.hist(ydata, bins = len(ydata), weights = weights, rwidth = .9, orientation = "horizontal")


# Finding average and error on the average per every strip

_data_statistics = [[key, stat(item)["mu"], stat(item)["stdDev"]] for key, item in data_dict.items()] # This is redefined better

# This is the actual reduced data
xdata = [stat(item)["mu"] for key, item in data_dict.items()]
ydata = ydata 
yerr = (np.ones_like(ydata) * ystep) / np.sqrt(12)
xerr = [stat(item)["stdDev"] / np.sqrt(len(item)) for key, item in data_dict.items()]

plt.figure(2)
plt.errorbar(xdata, ydata, yerr = yerr, xerr = xerr, linestyle = "None")


# Creating a single array of centered data
centered_data = np.array([])
chi2v = []
for key, item in data_dict.items():
    A = np.array(item - stat(item)["mu"])
    # print(A)
    bin_content, bin_edges, _ = plt.hist(A, bins = sturges(A), density = True)
    m = interp(bin_content, bin_edges, model_gauss_2)
    # chi2v.append(1. - chi2.cdf(m.fval, df = m.ndof))
    chi2v.append(m.fval / m.ndof)
    centered_data = np.concatenate((centered_data, A))

plt.figure(3)
plt.hist(chi2v, bins = sturges(chi2v), rwidth = .9, density = True)


# Plotting residuals


plt.legend()
plt.show()
