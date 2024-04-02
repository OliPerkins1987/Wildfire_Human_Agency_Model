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
        
        self.Fire_dat = {}
        self.Fire_vals= {}

        probs_key     = {'bool': 'yprob.TRUE', 
                         'ba'  : 'yval'} ### used for gathering final results
        
        
        ### gather numpy arrays 4 predictor variables
        x = 'arson'
            
        for b in ['bool', 'ba']:  
          
              
          if b in self.Fire_vars[x].keys(): 
           
            
                ### containers for fire outputs
                self.Fire_dat[x]       = {} if not x in self.Fire_dat.keys() else self.Fire_dat[x]  
                self.Fire_dat[x][b]    = []
                self.Fire_vals[x]      = {} if not x in self.Fire_vals.keys() else self.Fire_vals[x]  
                self.Fire_vals[x][b]   = []
            
                temp_key = self.Fire_vars[x][b]
    
    
                ### Gather relevant map data
                for y in range(len(temp_key)):
            
                    temp_val = self.model.p.Maps[temp_key[y]][self.model.timestep, :, :] if len(self.model.p.Maps[temp_key[y]].shape) == 3 else self.model.p.Maps[temp_key[y]]
            
                    self.Fire_dat[x][b].append(temp_val)

                ### combine predictor numpy arrays to a single pandas       
                self.Fire_dat[x][b]  = pd.DataFrame.from_dict(dict(zip(self.Fire_vars[x][b], 
                           [z.reshape(self.model.p.xlen*self.model.p.ylen).data for z in self.Fire_dat[x][b]])))
        
        
    ####################################
                
    ### Make predictions
                
    ####################################
                
                ##########
                ### Tree
                ##########
                
                if self.Fire_use[x][b]['type'] == 'tree_mod':
      
        
                    Fire_struct = define_tree_links(self.Fire_use[x][b]['pars'])

                    self.Fire_vals[x][b] = predict_from_tree_fast(dat =  self.Fire_dat[x][b], 
                              tree = self.Fire_use[x][b]['pars'], struct = Fire_struct, 
                               prob = probs_key[b], skip_val = -3.3999999521443642e+38, na_return = 0)
    
                ################
                ### Regression
                ################
                
                elif self.Fire_use[x][b]['type'] == 'lin_mod':
    
                    self.Fire_vals[x][b] = deepcopy(self.Fire_dat[x][b])
                    
                    ### Mulitply data by regression coefs
                    for coef in self.Fire_vars[x][b]:
                        
                        self.Fire_vals[x][b][coef] = self.Fire_vals[x][b][coef] * self.Fire_use[x][b]['pars']['coef'].iloc[np.where(self.Fire_use[x][b]['pars']['var'] == coef)[0][0]]
                    
                    ### Add intercept
                    self.Fire_vals[x][b] = self.Fire_vals[x][b].sum(axis = 1) + self.Fire_use[x][b]['pars']['coef'].iloc[np.where(self.Fire_use[x][b]['pars']['var'] == 'Intercept')[0][0]]
                    
                    ### Link function
                    self.Fire_vals[x][b] = regression_transformation(regression_link(self.Fire_vals[x][b], 
                                                  link = self.Fire_use[x][b]['pars']['link'][0]), 
                                                                     transformation = self.Fire_use[x][b]['pars']['transformation'][0])
                    
                    ### Use threshold, default at 0.5 - again defined empirically
                    self.Fire_vals[x][b] = pd.Series([x if x >= self.model.p.Constraint_pars['Arson_threshold'] else 0 for x in self.Fire_vals[x][b]])

        ### calculate burned area through DT and LM combination
        self.Fire_vals = (self.Fire_vals[x]['bool'] + self.Fire_vals[x]['ba']) / 2
        
        
    ###############################################################
                
    ### Adjust arson ignitions for degree of transition phase
                
    ###############################################################
    
        afr_vals = []
    
        ### Assume Nonex doesn't commit arson
        for ls in ['Cropland', 'Pasture', 'Rangeland', 'Forestry']:
        
            if 'Trans' in self.model.LFS[ls].keys():
                
                afr_vals.append(self.model.LFS[ls]['Trans'])
        
        ### remove agroforestry from logging contribution to arson
        if 'Agroforestry' in self.model.AFT_scores.keys():
            
            afr_vals.append(0 - self.model.AFT_scores['Agroforestry'])
        
        afr_vals = np.nansum(afr_vals, axis = 0)        
        afr_vals = pd.Series([x if x > 0 else 0 for x in np.array(afr_vals).reshape(self.model.p.xlen*self.model.p.ylen)])
        
        ### Multiply by regression of n-ignitions against Transition AFR
        self.Fire_vals = self.Fire_vals * np.exp(-2.184 + afr_vals * 1.166) 
        
        ### adjust for land area of pixel
        self.Fire_vals = self.Fire_vals * self.model.p.Maps['Mask']
        
        ### !! Add ajustment for ice
        
        
    ################################################################
    
    ### Add constraints
    
    ################################################################
    
        for c in self.model.Observers.values():
            
            if 'ct' in type(c[0]).__name__:
                
                c.constrain_arson()
        
