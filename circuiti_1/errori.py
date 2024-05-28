import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy.stats import chi2
import pandas as pd

# import sys
# sys.path.append("/home/yzemp/Documents/Programming/lab2")

##########################################################
# vars

sheet_id = "1TDCDWIADfjJNye4wf9-moEf_tEwMaD4bmi66gE59k80"
sheet_name = "uno"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data = pd.read_csv(url)


##########################################################
# models


##########################################################
# interpolations


#####################################################################
# Runtime

def main():
    pass



if __name__ == "__main__":
    main()
