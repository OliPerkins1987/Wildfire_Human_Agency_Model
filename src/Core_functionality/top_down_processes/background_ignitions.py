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
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_numpy
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.prediction_tools.predict_funs import get_LU_dat, predict_LU_behaviour


###########################################################################################

### Background rate class

###########################################################################################


class background_rate(AFT):
    
    def setup(self):
        AFT.setup(self)
        
        self.Fire_use = {'background_rate':{'bool': 'tree_mod', 
                                   'ba': 'lin_mod', 
                                   'size': np.nan}}
        
    def ignite(self):
    
        fire_dat = {}
        self.Fire_vals= {}

        probs_key     = {'ba'    : 'yval', 
                         'bool'  : 'yval'} ### used for gathering final results
        
        
        ### gather numpy arrays 4 predictor variables
        for x in self.Fire_use.keys():
            
            fire_dat[x]        = get_LU_dat(self, probs_key, self.Fire_vars[x])
            self.Fire_vals[x]  = predict_LU_behaviour(self, probs_key, 
                                 self.Fire_vars[x], fire_dat[x], 
                                 self.Fire_use[x], remove_neg= False, normalise = False)
        
        self.Fire_vals[x].update((x, np.array(y).reshape(self.model.ylen, 
                                   self.model.xlen)) for x, y in self.Fire_vals[x].items())
        
        ### polynomial correction
        self.Fire_vals[x]['ba'] = self.Fire_vals[x]['ba'] + ((self.model.p.Maps['ET'][self.model.timestep, :, :].data**2)*-1.310211e-04)
        self.Fire_vals[x]['ba'] = self.Fire_vals[x]['ba'] + ((self.model.p.Maps['MA'][self.model.timestep, :, :].data**2)*-5.212816e+02)
        self.Fire_vals[x]['ba'] = np.exp(self.Fire_vals[x]['ba'])
        self.Fire_vals[x]['ba'] = np.select([self.Fire_vals[x]['ba'] > 0], [self.Fire_vals[x]['ba']], default = 0)
            
        ### calculate burned area through bool & ba%
        self.Fire_vals = np.array(list(self.Fire_vals[x].values())).mean(axis = 0)      
        
        ### adjust for land area of pixel
        self.Fire_vals = self.Fire_vals * self.model.p.Maps['Mask'].reshape(self.model.ylen, self.model.xlen)
        
       
