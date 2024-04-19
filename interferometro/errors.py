import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

import sys
sys.path.append("../")
import my_stats


##########################################################3
# vars

sheet_id = "1dRjk3ARX3-TDBIuWrPaqR57W10_ucC5kEYAa6Lzmtn0"
sheet_name = "errors"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)

errors = data["Errors"].to_numpy()

#####################################################################
# Runtime

def main():
    plt.hist(errors, density = False, bins = 5)
    print(my_stats.stat(errors))
    errors.sort()
    print(f"Std dev (uniforme):\t{(errors[-1] - errors[0]) / np.sqrt(12)}")


    plt.show()
    

if __name__ == "__main__":
    main()
