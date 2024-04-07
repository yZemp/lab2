import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

import sys
sys.path.append("/home/yzemp/Documents/Programming/lab2")
# import error_prop_bolde as errpropb

##########################################################3
# vars

# data = pd.read_excel("https://docs.google.com/spreadsheets/d/1bjtqJHRvWQMS7QxDNb8iBOs0CKm75ioh5B47gZAaU2Y/export?format=xlsx", sheet_name = None)
# print(data["gradi"])

alpha = 1.043464486

sheet_id = "1bjtqJHRvWQMS7QxDNb8iBOs0CKm75ioh5B47gZAaU2Y"
sheet_name = "spettro_hg_prisma"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

lambdas = data["Mao3 (boh)"].to_numpy()
print(lambdas)

deltams = data["radianti"].to_numpy()
print(deltams)

ns = [np.sin((deltam + alpha) / 2) / np.sin(alpha / 2) for deltam in deltams]


def propagation(delta):
    return np.power(np.cos((delta + alpha) / 2) / (2 * np.sin(alpha / 2)), 2) + np.power((np.sin(alpha + delta / 2)) / (np.cos(alpha) - 1), 2)


errors = [np.sqrt(propagation(delta)) * .0008 for delta in deltams]
# errors_2 = [errpropb.propagazione_errore(["delta", "alpha"], "sin((deltam + alpha) / 2) / sin(alpha / 2)", [delta, alpha], [[1, 0], [0, 1]], [], []) for delta in deltams]
# print(errors, errors_2)

##########################################################3
# models

def cauchy(lambd, A, B):
    return A + B / np.power(lambd, 2)


##########################################################3
# interpolations

def interp(x, y, yerr, func = cauchy):
    my_cost = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(my_cost, 1, 1)
    m.migrad()
    m.hesse()
    return m




#####################################################################
# Runtime

def main():

    plt.errorbar(lambdas, ns, errors, label = "Linee di emissione", linestyle = "", marker = "o", c = "#151515")

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(lambdas, ns, errors)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    lnsp = np.linspace(lambdas[0] - 10, lambdas[-1] + 10, 10_000)
    plt.plot(lnsp, cauchy(lnsp, *m1.values), label = "Legge di Cauchy", c = "#a515d5")

    plt.xlabel("Lunghezza d\'onda [$\lambda$]")
    plt.ylabel("Indice di rifrazione")

    plt.plot([], [], ' ', label = f"P-value: {1. - chi2.cdf(m1.fval, df = m1.ndof):.4f}")
    plt.plot([], [], ' ', label = f"A = {m1.values[0]:.3f} $\pm$ {m1.errors[0]:.3f}")
    plt.plot([], [], ' ', label = f"B = 9700 $\pm$ 500")
    # plt.plot([], [], ' ', label = f"B = {m1.values[1]:.0f} $\pm$ {m1.errors[1]:.0f}")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
