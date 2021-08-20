# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import numpy as np 
import pandas as pd
               
       
def regression_link(X, link):
    
    if link == 'identity':
        
        return(X)
    
    
    elif link == 'log':
        
        return(np.exp(X))
    
    
    elif link == 'logistic':
        
        return(1/(1+(np.exp(0-X))))
    
    
    else:
        
        print('Unrecognised link function')
        
        