import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd
from error_prop_bolde import *

#####################################################################
# data

alpha = 1.043464486
A = 1.592
B = 9703

sheet_id = "1bjtqJHRvWQMS7QxDNb8iBOs0CKm75ioh5B47gZAaU2Y"
sheet_name = "spettro_hg_prisma"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)


def angolo_to_lambda(delta):
    n = np.sin((delta + alpha) / 2) / np.sin(alpha / 2)
    lambd = np.sqrt(B / (n - A))
    return lambd

angoli_ignoto = data["radianti"].to_numpy()
lambdas = [angolo_to_lambda(delta) for delta in angoli_ignoto]
print(lambdas)


def propagation(delta):
    return np.power(np.cos((delta + alpha) / 2) / (2 * np.sin(alpha / 2)), 2) + np.power((np.sin(alpha + delta / 2)) / (np.cos(alpha) - 1), 2)

lambda_errors = [np.sqrt(propagation(delta)) * .0008 for delta in angoli_ignoto]




#####################################################################
# Runtime

def main():

    plt.errorbar(lambdas, np.ones_like(lambdas) * 0, xerr = lambda_errors, label = "Linee di emissione", linestyle = "", marker = "o", c = "#151515")

    

    plt.xlabel("Lunghezza d\'onda [$\lambda$]")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
