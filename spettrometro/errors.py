import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2

import sys
sys.path.append("../")
import my_stats


##########################################################3
# vars

errors = [5,3,1,-2,-5,3,-4,2,-1]

#####################################################################
# Runtime

def main():
    plt.hist(errors, density = False, bins = my_stats.sturges(errors))
    print(my_stats.stat(errors))
    errors.sort()
    print(f"Std dev (uniforme):\t{(errors[-1] - errors[0]) / np.sqrt(12)}")


    plt.show()
    

if __name__ == "__main__":
    main()
