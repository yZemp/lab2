import numpy as np
#from random import random
import scipy as sp
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


if __name__ == "__main__":
        print(sturges([i for i in range(1, 1484)]))
        print(sturges(1484))
