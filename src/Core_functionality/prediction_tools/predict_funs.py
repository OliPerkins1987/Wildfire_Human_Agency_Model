# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:34:47 2024

@author: Oli
"""

import agentpy as ap
import numpy as np
import pandas as pd
from copy import deepcopy

from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation, variable_transformation
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast

def get_LU_dat(aft, probs_dict, vars_dict):
    
    ''' takes an aft and returns a set of land use behaviour parameters'''
    ''' behaviour dict should be a dict with two keys: type and vars'''
    ''' use type key to stack multiple models for a single land use behaviour'''
    
    dat = {}
       
    for b in probs_dict.keys():  
          
        if b in vars_dict.keys():
            
            if vars_dict[b] != 'constant':
            
                ### containers for data
                dat[b]   = []
                temp_key = vars_dict[b]
        
                ### Gather relevant map data
                for y in range(len(temp_key)):
            
                    temp_val = aft.model.p.Maps[temp_key[y]][aft.model.timestep, :, :] if len(aft.model.p.Maps[temp_key[y]].shape) == 3 else aft.model.p.Maps[temp_key[y]]
            
                    dat[b].append(temp_val.data)

                ### combine predictor numpy arrays to a single pandas       
                dat[b]  = pd.DataFrame.from_dict(dict(zip(vars_dict[b], 
                          [z.reshape(aft.model.p.xlen*aft.model.p.ylen).data for z in dat[b]])))
        
            else:
            
                dat[b] = 'None'
        
    return(dat)
        
    ####################################
                
    ### Make predictions
                
    ####################################

def predict_LU_behaviour(aft, probs_dict, vars_dict, dat, pars, 
                         skip_thresh = -1e+10, remove_neg = True, normalise = True):
    
    vals = {}
       
    for b in probs_dict.keys():  
          
        if b in vars_dict.keys(): 
    
            if 'constant' in pars[b].keys():              
                  
                vals[b] = pd.Series([pars[b]['constant']] * (aft.model.p.ylen * aft.model.p.xlen))
                
                ### mask for land areas
                vals[b] = pd.Series(aft.p.Maps['Mask'] >0) * vals[b]        
    
            elif pars[b]['type'] == 'tree_mod':
                
                struct = define_tree_links(pars[b]['pars'])

                vals[b]= predict_from_tree_fast(dat =  dat[b], 
                              tree = pars[b]['pars'], struct = struct, 
                               prob = probs_dict[b], skip_val = -1e+10, na_return = 0)
    
                ################
                ### Regression
                ################
                
            elif pars[b]['type'] == 'lin_mod':
    
                vals[b] = deepcopy(dat[b]).astype(float)
                vals[b] = vals[b].where(vals[b] > skip_thresh, 0)
                
                ### Apply any transformations and mulitply data by regression coefs
                for coef in vars_dict[b]:
                    
                    vals[b][coef] = variable_transformation(vals[b][coef], pars[b]['pars']['var_trans'].iloc[np.where(pars[b]['pars']['var'] == coef)[0][0]])
                    vals[b][coef] = vals[b][coef] * pars[b]['pars']['coef'].iloc[np.where(pars[b]['pars']['var'] == coef)[0][0]]
                    
                ### Add intercept
                vals[b] = vals[b].sum(axis = 1) + pars[b]['pars']['coef'].iloc[np.where(pars[b]['pars']['var'] == 'Intercept')[0][0]]
                    
                ### Link function
                vals[b] = regression_transformation(regression_link(vals[b], 
                                                  link = pars[b]['pars']['link'][0]), 
                                                  transformation = pars[b]['pars']['transformation'][0])
                    
                ### control for negative values
                if remove_neg == True:
                    vals[b] = pd.Series([x if x > 0 else 0 for x in vals[b]])
                
                ### control for output probs / coverage > 1
                if normalise == True:
                    vals[b] = pd.Series([x if x < 1 else 1 for x in vals[b]])
                     
                ### mask for land areas
                vals[b] = pd.Series(aft.p.Maps['Mask'] >0) * vals[b]
                                
                
        else:
                        
            pass
         
        
    return(vals)
            
    
    
    



