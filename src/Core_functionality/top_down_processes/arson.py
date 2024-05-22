# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np
from copy import deepcopy

from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.prediction_tools.predict_funs import get_LU_dat, predict_LU_behaviour


###########################################################################################

### Arson class

###########################################################################################


class arson(AFT):
    
    def setup(self):
        AFT.setup(self)
        
        self.Fire_use = {'arson':{'bool': 'tree_mod', 
                                   'ba': 'lin_mod', 
                                   'size': np.nan}}
        
    def ignite(self):

       
    ####################################
                
    ### Prepare data
                
    ####################################
        
        fire_dat = {}
        self.Fire_vals= {}

        probs_key     = {'bool': 'yprob.TRUE', 
                         'ba'  : 'yval'} ### used for gathering final results
        
       ### gather numpy arrays 4 predictor variables
        for x in self.Fire_use.keys():
            
            fire_dat[x]        = get_LU_dat(self, probs_key, self.Fire_vars[x])
            self.Fire_vals[x]  = predict_LU_behaviour(self, probs_key, 
                                 self.Fire_vars[x], fire_dat[x], 
                                 self.Fire_use[x], remove_neg= False, normalise = False)
        
        self.Fire_vals[x].update((x, np.array(y).reshape(self.model.ylen, 
                                   self.model.xlen)) for x, y in self.Fire_vals[x].items())
                    
        ### calculate burned area through DT and LM combination
        self.Fire_vals[x] = np.array(list(self.Fire_vals[x].values())).mean(axis = 0)     
        
        
    ###############################################################
                
    ### Adjust arson ignitions for specific land use conflicts
                
    ###############################################################
    
        afr_vals = []
    
        ### Assume Nonex doesn't commit arson
        for ls in ['Cropland', 'Pasture', 'Rangeland', 'Forestry']:
        
            if 'Trans' in self.model.LFS[ls].keys():
                
                afr_vals.append(self.model.LFS[ls]['Trans'])
                
            if 'Pre' in self.mode.LFS[ls].keys():
                
                afr_vals.append(self.model.LFS[ls]['Pre'])
        
        ### remove agroforestry from logging contribution to arson
        if 'Agroforestry' in self.model.AFT_scores.keys():
            
            afr_vals.append(0 - self.model.AFT_scores['Agroforestry'])
        
        
        afr_vals = np.nansum(afr_vals, axis = 0)
        
        ### adjust for protected areas
        if 'Conservationist' in self.model.AFT_scores.keys():
            
            afr_vals = afr_vals * self.model.AFT_scores['Conservationist']
        
        afr_vals = np.select([afr_vals > 0], [afr_vals], default = 0)
        
        ### Multiply by regression of n-ignitions against Transition AFR
        self.Fire_vals = self.Fire_vals * (afr_vals*6.548657 - 0.009306)
        
        ### adjust for land area of pixel
        self.Fire_vals = self.Fire_vals * self.model.p.Maps['Mask']
        
        
    ################################################################
    
    ### Add constraints
    
    ################################################################
    
        for c in self.model.Observers.values():
            
            if 'ct' in type(c[0]).__name__:
                
                c.constrain_arson()
        
