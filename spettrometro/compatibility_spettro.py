import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd


alpha = 1.043464486
A = 1.592
B = 9703

sheet_id = "1bjtqJHRvWQMS7QxDNb8iBOs0CKm75ioh5B47gZAaU2Y"
sheet_name = "spettro_ignoto_prisma"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)


# angoli_ignoto = data["radianti"].to_numpy()
lambdas = data["Lambda"].to_numpy()
lambda_errors = data["Sigma lambda"].to_numpy()
veri = [404.6563,407.7837,433.9223,485.5584,491.6068,546.0735,576.9598,579.0663, 629.1228, 671.634]

def chi2(x, xv, errs):
    if len(x) != len(xv) or len(xv) != len(errs): return -1
    sum = 0
    for i in range(len(x)):
        sum += abs(x[i] - xv[i]) / errs[i]
    
    return sum

chi2 = chi2(lambdas, veri, lambda_errors)
print(chi2)
