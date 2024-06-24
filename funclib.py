import numpy as np


# Edi - BUGGED
# def max_sez_aurea(func, xmin, xmax, *args, prec = .0001):
#    '''func of type f(x, args)'''
#    
#    r = ( - 1 + np.sqrt(5)) / 2  #golden ratio
#
#    while abs(xmax - xmin) > prec:
#
#        a = xmin +      r * abs(xmax - xmin)
#        b = xmin + (1 - r) * abs(xmax - xmin)
#        
#        if func(b, *args) < func(a, *args):
#            xmin = b
#        else: 
#            xmax = a
#
#    return xmin, func(xmin, *args)



##############################################################
# Zero


def find_zero(
    g,              # funzione di cui trovare lo zero
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione
    '''
    xAve = xMin 
    while ((xMax - xMin) > prec) :
        xAve = 0.5 * (xMax + xMin) 
        if (g (xAve) * g (xMin) > 0.): xMin = xAve 
        else                         : xMax = xAve 
    return xAve 
    


def find_zero_recursive(
    g,              # funzione di cui trovare lo zero  
    xMin,           # minimo dell'intervallo            
    xMax,           # massimo dell'intervallo          
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione ricorsivo
    '''
    xAve = 0.5 * (xMax + xMin)
    if ((xMax - xMin) < prec): return xAve ;
    if (g (xAve) * g (xMin) > 0.): return find_zero(g, xAve, xMax, prec) ;
    else                         : return find_zero(g, xMin, xAve, prec) ;
    



##############################################################
# MIN


def find_min_goldenratio(
    g,              # funzione di cui trovare il minimo
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    '''

    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g (x3) > g (x2)): 
            x0 = x3
            x1 = x1         
        else :
            x1 = x2
            x0 = x0          
            
        larghezza = abs (x1-x0)             
                                   
    return (x0 + x1) / 2. 



def find_min_goldenratio_recursive(
    g,              # funzione di cui trovare il minimo
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente
    '''

    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0) 
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return ( x0 + x1) / 2.
    elif (g (x3) > g (x2)) : return find_min_goldenratio_recursive(g, x3, x1, prec)
    else                   : return find_min_goldenratio_recursive(g, x0, x2, prec)   



##############################################################
# Max

def find_max_goldenratio(
    g,              # funzione di cui trovare il massimo
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    '''

    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g (x3) < g (x2)): 
            x0 = x3
            x1 = x1         
        else :
            x1 = x2
            x0 = x0          
            
        larghezza = abs (x1-x0)             
                                   
    return (x0 + x1) / 2. 



def find_max_goldenratio_recursive(
    g,              # funzione di cui trovare il massimo
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente
    '''

    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0) 
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return ( x0 + x1) / 2.
    elif (g (x3) < g (x2)) : return find_max_goldenratio_recursive(g, x3, x1, prec)
    else                   : return find_max_goldenratio_recursive(g, x0, x2, prec)   









if __name__ == '__main__':

    def f(x, a, b):
        return np.sin(a * x) + b

    x, y = max_sez_aurea(f, 0, np.pi, np.pi, 0)
    print(x, y)
