# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:34:47 2024

@author: Oli
"""

import agentpy as ap
import numpy as np
import pandas as pd


def get_LU_pars(behaviour_dict):
    
    ''' takes an aft and returns a set of land use behaviour parameters'''
    ''' behaviour dict should be a dict with two keys: type and vars'''
    ''' use type key to stack multiple models for a single land use behaviour'''
    
    outdict = {}
    
    
    if behaviour_dict['type'] == 'lin_mod':
                
        outdict = [x for x in behaviour_dict['pars'].iloc[:,0].tolist() if x != 'Intercept']      
                
    elif behaviour_dict['type'] == 'tree_mod':
                    
        outdict = [x for x in behaviour_dict['pars'].iloc[:,1].tolist() if x != '<leaf>']
    
    return(outdict)
            
    
    
    



