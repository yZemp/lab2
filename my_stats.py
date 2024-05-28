import numpy as np
import matplotlib.pyplot as plt
#from random import random
import scipy as sp
import scipy.stats

from scipy.stats import t as tfunc

import sys

def stat(sample):

        mu = np.average(sample)
        var = np.var(sample)
        stdDev = np.std(sample) 
        skewness = sp.stats.skew(sample)
        kurtosis = sp.stats.kurtosis(sample)

        return {"mu": mu, "var": var, "stdDev": stdDev, "sk": skewness, "ku": kurtosis}

def sturges(N):
        '''returns the surges function applied to the sample length'''

        try:
                N = len(N)
        except:
                pass
        
        finally:
                return int(np.ceil(1 + 3.322 * np.log(N)))

def t_test(t, ndof):
        '''
        Pass t-value and ndof to find p-value  
        '''
        return 1 - tfunc.cdf(t, ndof) + tfunc.cdf(-t, ndof)



if __name__ == "__main__":
        l = np.linspace(-10, 10, 10_000)
        plt.plot(l, tfunc.cdf(l, 1))
        plt.show()

