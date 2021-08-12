# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:03:42 2021

@author: Oli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_model_output(model, kind, ls_type):

    if kind == 'AFT':    

        temp = np.column_stack([(x.reshape(27648))for x in model.AFT_scores.values()])
        temp = pd.DataFrame(temp)
        
    elif kind == 'LFS':
        
        temp = np.column_stack([(x.reshape(27648))for x in model.LFS[ls_type].values()])
        temp = pd.DataFrame(temp)
        

    return(temp)


def get_afr_vals(afr_dict):
    
    afr_res = {}
    
    for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
        afr_vals = []
    
        for ls in afr_dict.keys():
        
            if afr in afr_dict[ls].keys():
                
                afr_vals.append(afr_dict[ls][afr])
                
        afr_res[afr] = sum(afr_vals)
    
    return(afr_res)