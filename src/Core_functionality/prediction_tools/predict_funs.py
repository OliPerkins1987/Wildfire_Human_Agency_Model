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
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_numpy

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
                dat[b]  = np.array([x.reshape(
                           aft.model.p.xlen*aft.model.p.ylen).data for x in dat[b]]).transpose()
        
        
        
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
                  
                vals[b] = np.array([pars[b]['constant']] * (aft.model.p.ylen * aft.model.p.xlen))
                
                ### mask for land areas
                vals[b] = np.array(aft.p.Maps['Mask'] >0) * vals[b]        
    
            elif pars[b]['type'] == 'tree_mod':
                
                struct = define_tree_links(pars[b]['pars'])

                vals[b]= predict_from_tree_numpy(dat =  dat[b], 
                          tree = pars[b]['pars'], struct = struct, split_vars = vars_dict[b],
                          prob = probs_dict[b], skip_val = skip_thresh, na_return = 0)
    
                ################
                ### Regression
                ################
                
            elif pars[b]['type'] == 'lin_mod':
    
                vals[b] = deepcopy(dat[b]).astype(float)
                vals[b] = np.select([vals[b] > skip_thresh], [vals[b]], default = 0)
                
                ### Apply any transformations and mulitply data by regression coefs
                for coef in range(len(vars_dict[b])):
                    
                    vals[b][:, coef] = variable_transformation(vals[b][:, coef], pars[b]['pars']['var_trans'].iloc[np.where(pars[b]['pars']['var'] == vars_dict[b][coef])[0][0]])
                    vals[b][:, coef] = vals[b][:, coef] * pars[b]['pars']['coef'].iloc[np.where(pars[b]['pars']['var'] == vars_dict[b][coef])[0][0]]
                    
                ### Add intercept
                vals[b] = vals[b].sum(axis = 1) + pars[b]['pars']['coef'].iloc[np.where(pars[b]['pars']['var'] == 'Intercept')[0][0]]
                    
                ### Link function
                vals[b] = regression_transformation(regression_link(vals[b], 
                                                  link = pars[b]['pars']['link'][0]), 
                                                  transformation = pars[b]['pars']['transformation'][0])
                    
                ### control for negative values
                if remove_neg == True:
                    vals[b] = np.select([vals[b] > 0], [vals[b]], default = 0)
                
                ### control for output probs / coverage > 1
                if normalise == True:
                    vals[b] = np.select([vals[b] < 1], [vals[b]], default = 1)
                     
                ### mask for land areas
                vals[b] = np.select([aft.p.Maps['Mask'] >0], [vals[b]], default = 0)
                                
                
        else:
                        
            pass
         
        
    return(vals)
            
    
    
    



