from IPython import display
from IPython.display import Latex
from sympy import *
import numpy as np

def err_prop_display(variables, formula):
    '''
    Used in ERROR_PROPAGATION.
    Auxiliary function that displays the error propagation in Latex.
    Particular cases (one or two variables) are managed separately.
    '''

    if(len(variables)==1):
      # sigma
      sigma = symbols(f'sigma_{variables[0]}')
      # display
      display(Latex('\sigma=\\left|{}\\right|{}'.format(latex(diff(formula, variables[0])), latex(sigma))))
      return

    if(len(variables)==2):
      # sigmas
      sigmas = list()
      for ausy in variables:
          sigmas.append(f'sigma_{ausy}')
      sigmastring = ' '.join(sigmas)
      sigmas = symbols(sigmastring)

      # covariances
      covar = symbols(f'sigma_{variables[0]}{variables[1]}')

      exp = 0
      # add the sigmas
      for i in range(len(variables)):
          exp += (diff(formula, variables[i]))**2 * sigmas[i]**2
      # add the cov
      exp += 2 * (diff(formula, variables[0])) * (diff(formula, variables[1])) * covar
      # display
      display(Latex('\sigma='+latex(sqrt(exp))))

      return

    else:

      # sigmas
      sigmas = list()
      for ausy in variables:
          sigmas.append(f'sigma_{ausy}')
      sigmastring = ' '.join(sigmas)
      sigmas = symbols(sigmastring)

      # covariances
      covars = list()
      for i in range(len(variables)):
        for j in range(len(variables)-i-1): # ideally the covariance matrix is symmetrical: we use only the upper triangular
          covars.append(f'sigma_{variables[i]}{variables[1+j+i]}')
      covarstring = ' '.join(covars)
      covars = symbols(covarstring)

      exp = 0
      # add the sigmas
      for i in range(len(variables)):
          exp += (diff(formula, variables[i]))**2 * sigmas[i]**2
      # add the covs
      k = 0 # counter to go trough the covariance symbols vector
      for i in range(len(variables)):
        for j in range(len(variables)-i-1): # ideally the covariance matrix is symmetrical: we use only the upper triangular
          exp += 2 * (diff(formula, variables[i])) * (diff(formula, variables[1+j+i])) * covars[k]
          k += 1

      # display
      display(Latex('\sigma='+latex(sqrt(exp))))

      return


def error_propagation(vector, formula, values, cov_matr, display = True):
    '''
    Function to evaluate the propagated uncertainty of a quantity obtained with the expression in FORMULA (str) from the other quantities in VECTOR (list of str);
    in the point of evaluation the quantities have the values in VALUES (list of float) and the covariance matrix is COV_MATR (list of list of float).
    The expression of propagation is displayed only when display=True (default).
    '''

    # raise errors
    if(len(vector) != len(values)): raise ValueError('variables and values mismatch')
    if((len(vector) != len(cov_matr))): raise ValueError('variables and errors mismatch')
    if(len(cov_matr) > 1):
      for i in range(len(cov_matr)):
        if((len(vector) != len(cov_matr[i]))): raise ValueError('variables and errors mismatch')

    # create list of variables from names in VECTOR
    if(len(vector) == 1): variables = vector
    else: variables = list(symbols(' '.join(vector)))

    if(display): err_prop_display(variables, formula)

    derivatives = []
    # create derivative expressions vector
    for i in range(len(vector)): derivatives.append(str(diff(formula, variables[i])))
    # substitute variable names with value names
    for i in range(len(derivatives)):
      for j in range(len(variables)):
        derivatives[i] = derivatives[i].replace(str(variables[j]), 'values[{}]'.format(j))
    # evaluates the derivatives
    for i in range(len(derivatives)): derivatives[i] = float(eval(derivatives[i]))

    # final matrix product
    if(len(derivatives)==1): final_error = np.sqrt(derivatives[0]**2 * cov_matr[0])
    else:
      derivatives = np.array(derivatives)
      final_error = np.sqrt(np.dot(derivatives.T, np.dot(cov_matr, derivatives)))

    return final_error