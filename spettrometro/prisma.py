import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

##########################################################3
# vars

# data = pd.read_excel("https://docs.google.com/spreadsheets/d/1bjtqJHRvWQMS7QxDNb8iBOs0CKm75ioh5B47gZAaU2Y/export?format=xlsx", sheet_name = None)
# print(data["gradi"])

alpha = 1.043464486

# lambdas = [404.6563,407.7837,433.9223,434.7494,435.8328,546.0735,576.9598,579.0663]
# Guesswork sui verde acqua:
# lambdas = [404.6563,407.7837,433.9223,485.5584,491.6068,546.0735,576.9598,579.0663]
# lambdas = [404.6563,407.7837,433.9223,546.0735,576.9598,579.0663]

# deltams = [0.8914269155,0.889390698,0.8766643389,0.85856,0.85594,0.8451756903,0.8395760923,0.8384125394]
# deltams = [0.8914269155,0.889390698,0.8766643389,0.8451756903,0.8395760923,0.8384125394]


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

    plt.errorbar(lambdas, ns, errors, label = "Data", linestyle = "", marker = "o", c = "#050505")

    print("----------------------------------------------- M1 -----------------------------------------------")
    m1 = interp(lambdas, ns, errors)
    print(m1.migrad())
    print(f"Pval:\t{1. - chi2.cdf(m1.fval, df = m1.ndof)}")
    
    lnsp = np.linspace(lambdas[0], lambdas[-1], 10_000)
    plt.plot(lnsp, cauchy(lnsp, *m1.values), label = "Cauchy")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
