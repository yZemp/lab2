import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
# from error_prop_bolde import *

#####################################################################
# data

lambdas = [484.0, 552.7, 569.5, 572.2]
lambda_errors = np.ones_like(lambdas) * 2

lveri = [498.2, 568.8, 589.0, 589.6]

#####################################################################
# Runtime

def main():

    plt.errorbar(lambdas, [i % 2 for i in range(len(lambdas))], xerr = lambda_errors, capsize = 5, label = "Linee di emissione", linestyle = "", marker = "d", c = "#151515")
    # plt.errorbar(lambdas, [-2 for i in range(len(lambdas))], xerr = lambda_errors, capsize = 5, label = "Linee di emissione", linestyle = "", marker = "d", c = "#151515")
    plt.scatter(lveri, [-2 for i in range(len(lveri))], label = "Dati NIST", linestyle = "", marker = "d", c = "#157015")

    
    plt.xlabel("Lunghezza d\'onda [$\lambda$]")
    plt.ylim(-8, 12)

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
