import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd
# from error_prop_bolde import *

#####################################################################
# data

alpha = 1.043464486
A = 1.592
B = 9703

sheet_id = "1bjtqJHRvWQMS7QxDNb8iBOs0CKm75ioh5B47gZAaU2Y"
sheet_name = "spettro_ignoto_reticolo"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)


# angoli_ignoto = data["radianti"].to_numpy()
lambdas = data["Lambda"].to_numpy()
lambda_errors = data["Sigma lambda2"].to_numpy()

#####################################################################
# Runtime

def main():

    plt.errorbar(lambdas, [i % 2 for i in range(len(lambdas))], xerr = lambda_errors, capsize = 5, label = "Linee di emissione", linestyle = "", marker = "d", c = "#151515")
    # plt.errorbar(lambdas, [0 for i in range(len(lambdas))], xerr = lambda_errors, capsize = 5, label = "Linee di emissione", linestyle = "", marker = "d", c = "#151515")

    
    plt.xlabel("Lunghezza d\'onda [$\lambda$]")
    plt.ylim(-8, 12)

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
