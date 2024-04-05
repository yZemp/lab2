from IPython import display
from IPython.display import Latex
from sympy import *
import numpy as np


def insert( vector ) -> list:
    #Handling edge cases
    if len(vector) == 1:
      vector = symbols( str(vector) )
      sigmas = f"sigma_{vector}"
      covar = None
      all = sigmas
    elif len(vector) == 2:
      string = ' '.join( vector )
      vector = symbols( string )
    #creation of sigmas and covariants
      sigmas = []
      covar = []
      all = []
      for i in range(len(vector)):
        sigmas.append( f"sigma_{vector[i]}" )
        all.append( f"sigma_{vector[i]}" )
      sigmastring = ' '.join( sigmas )
      sigmas = symbols( sigmastring )
      covar.append(f"sigma_{vector[0]}{vector[1]}")
      covarstring = ' '.join( covar )
      covar = symbols( covarstring )
      all.append(covar)
    else:
      string = ' '.join( vector )
      vector = symbols( string )
      #creation of sigmas and covariants
      sigmas = []
      covar = []
      all = []
      for i in range(len(vector)):
        for j in range( i , len(vector)):
          if i == j:
            sigmas.append( f"sigma_{vector[i]}" )
            all.append( f"sigma_{vector[i]}" )
          else:
            covar.append( f"sigma_{vector[i]}{vector[j]}")
            all.append( f"sigma_{vector[i]}{vector[j]}")

      sigmastring = ' '.join( sigmas )
      sigmas = symbols( sigmastring )
      covarstring = ' '.join( covar )
      covar = symbols( covarstring )
    return vector, sigmas , covar, all

def derivazione(variables, formula, sigmas , covar) -> str:
    expo = 0
    # add the sigmas
    for i in range(len(variables)):
      expo += (diff(formula, variables[i]))**2 * sigmas[i]
      # add the covs
    k = 0
    if isinstance( covar , Symbol) :
      expo += 2 * (diff(formula, variables[0])) * (diff(formula, variables[1])) * covar
    else:
      for i in range(len(variables)):
        for j in range(len(variables)-i-1):
          expo += 2 * (diff(formula, variables[i])) * (diff(formula, variables[1+j+i])) * covar[k]
          k += 1
    return expo



def propagazione_errore (vector, formula , values , covmat , var_else = None, val_else = None, Display = True) -> str:

    if isinstance(var_else , list):
      things,a,a,a = insert(var_else)

    variables,sigmas,covar,all = insert(vector)

    expo = derivazione (variables, formula, sigmas , covar)

    if Display: display(Latex('\sigma='+latex(simplify(sqrt(expo)))))

    if len(variables) == 1:
      expo = expo.subs( variables , values)
      expo = expo.subs( sigmas , covmat)
    else:
      for i in range(len(variables)): expo = expo.subs( variables[i] , values[i])
      k = 0
      for i in range(len(covmat)):
        expo = expo.subs( all[k] , covmat[i][i])
        k +=1
      for i in range( len(covmat)):
        for j in range(i+1 , len(covmat[i])):
          expo = expo.subs( all[k] , covmat[i][j])
          k+=1
    if isinstance(var_else , list):
      if len(var_else ) == 1:
        expo = expo.subs( symbols(var_else[0]) , val_else[0])
      else:
        for i in range(len(var_else)): expo = expo.subs( things[i] , val_else[i])
    return sqrt(expo)




#######################################################################################################
# Istruzioni

# vector = ['a' , 'b' , 'c']
# formula = 'a+b*x+c*x^2'
# values = [1,2,3]
# covs = [[0,1,2],[1,2,3],[4,2,1]]
# var = ['x' , 'y']
# variues = [1.55 , 0.5]

# per utilizzare la funzione, inserire:
# - vector , contenente le variabili da derivare sottoforma di stringhe
# - formula, la formula in sintassi di python, non preoccuparti di importare le funzioni dalle librerie
# - values, vettore con i valori delle variabili da derivare
# - covs, matrice covarianza
# - var, le variabili inserite nella funzione non da derivare
# - variues, valori delle variabili non da derivare
# - display (facoltativa), variabile Booleana per avere il display della formula o no
# (propagazione_errore(vector , formula , values , covs , var , variues))