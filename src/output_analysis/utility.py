# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:03:42 2021

@author: Oli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_model_output(model, kind):

    if kind == 'AFT':    

        temp = np.column_stack([(x.reshape(27648))for x in model.AFT_scores.values()])
        temp = pd.DataFrame(temp)
        temp.columns = [type(x).__name__ for x in model.agents]
        
    elif kind == 'LFS':
        
        temp = np.column_stack([(x.reshape(27648))for x in model.LFS.values()])
        temp = pd.DataFrame(temp)
        

    return(temp)


def get_afr_vals(afr_dict):
    
    afr_res = {}
    
    for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
        afr_vals = []
    
        for ls in afr_dict.keys():
        
            if afr in afr_dict[ls].keys():
                
                afr_vals.append(afr_dict[ls][afr])
                
        afr_res[afr] = np.nansum(afr_vals)
    
    return(afr_res)


def get_fire_maps(model):
        
    fout = {}
        
    for a in model.agents:
            
        fout[type(a).__name__] = a.Fire_vals
    
    fout = pd.DataFrame.from_dict(fout)
    
    return(fout)
        
            
def get_escape_fire(model):
    
    res = []
    
    for year in model.results['Escaped_fire']:
    
        res.append(pd.DataFrame.from_dict(dict(zip(year.keys(), 
        [x.reshape(27648) for x in year.values()]))))
    
    res = [np.array(x.sum(1)).reshape(144, 192) for x in res]
    
    return(res)


