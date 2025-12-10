# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:03:42 2021

@author: Oli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_model_output(model, kind):

    """
    Extract model output data organized by agent type. 
    Converts agent data into a labeled DataFrame with one column per 
    agent class for analysis and export.    
    """

    if kind == 'AFT':    

        temp = np.column_stack([(x.reshape(27648))for x in model.AFT_scores.values()])
        temp = pd.DataFrame(temp)
        temp.columns = [type(x).__name__ for x in model.agents]
        
    return(temp)


def get_afr_vals(afr_dict):
    
    """
    Aggregate fire regime values across anthropogenic fire regimes. 
    Sums distribution of anthropogenic fire regimes across land use systems 
    (Pre, Trans, Intense, Post) and a returns dictionary.
    """
    
    
    afr_res = {}
    
    for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
        afr_vals = []
    
        for ls in afr_dict.keys():
        
            if afr in afr_dict[ls].keys():
                
                afr_vals.append(afr_dict[ls][afr])
        
        afr_res[afr] = np.nansum(afr_vals, axis = 0)
        
    return(afr_res)


def get_fire_maps(model):
    
    """
    Create a DataFrame of fire values for each agent type across 
    the spatial grid. Returns pixel fraction burned by AFT.
    """
    
    fout = {}
        
    for a in model.agents:
            
        fout[type(a).__name__] = a.Fire_vals
    
    fout = pd.DataFrame.from_dict(fout)
    
    return(fout)
        
            
def get_escape_fire(model):
    
    """
    Extract escaped fire events by year and reshape to spatial grids. 
    Returns list of spatial arrays where each array represents total 
    escaped fire area for one year across the landscape.
    """
    
    
    res = []
    
    for year in model.results['Escaped_fire']:
    
        res.append(pd.DataFrame.from_dict(dict(zip(year.keys(), 
        [x.reshape(27648) for x in year.values()]))))
    
    res = [np.array(x.sum(1)).reshape(144, 192) for x in res]
    
    return(res)


