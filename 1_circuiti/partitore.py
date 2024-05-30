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
sheet_name = "partitore"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

x = data["Resistenza (kOhm)"].to_numpy()
sx = data["Sigma resistenza (kOhm)"].to_numpy()
y = data["Diff. potenziale (V)"].to_numpy()
sy = data["Sigma diff. potenziale (V)"].to_numpy()

# lnsp = np.linspace(0, x[-1] + 100, 100_000)
plt.errorbar(x, y, sy, xerr = sx,  linestyle = "None", c = "#090909", marker = "o", markersize = 3, alpha = 1, label = "Data")
plt.vlines(1, 0, y[-1] + .2, color = "#17c1d4", linestyle = "dotted")


plt.xscale("log")

plt.xlabel("Resistenza [kOhm]")
plt.ylabel("Tensione [V]")

# plt.plot(lnsp, )


plt.legend()
plt.show()