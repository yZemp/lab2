import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
# from error_prop_bolde import *

#####################################################################
# data

d1_fp = 0.0000181
derr1_fp = 0.0000003

d2_fp = 0.00000775
derr2_fp = 0.00000013

d1_m = 0.0000173
derr1_m = 0.0000006

d2_m = 0.0000081
derr2_m = 0.0000004

#####################################################################
# Runtime

def main():

    plt.errorbar([d1_fp, d2_fp], [1, 1], xerr = [derr1_fp, derr2_fp], capsize = 5, label = "Fabry-Perot", linestyle = "", marker = "o", c = "#660585")
    plt.errorbar([d1_m, d2_m], [0, 0], xerr = [derr1_m, derr2_m], capsize = 5, label = "Michelson", linestyle = "", marker = "o", c = "#056916")

    
    plt.xlabel("Distanza")
    plt.ylim(-8, 12)

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
