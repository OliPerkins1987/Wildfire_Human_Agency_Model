# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.prediction_tools.predict_funs import get_LU_dat, predict_LU_behaviour

###########################################################################################

### Arable AFTs

###########################################################################################

class Swidden(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Cropland'
        self.Habitat = {'Map': 'Cropland', 
                        'Constraint': 0.5776, 
                        'Constraint_type': 'lt'}
        
        self.Fire_use = {'cfp': {'bool': 'tree_mod', 
                                 'ba':   'lin_mod', 
                                 'size': 0.728455277}}
        
        self.Defor_size = 0.833333333


class SOSH(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Cropland'
        self.Habitat = {'Map': 'HDI_GDP', 
                        'Constraint': 7.714, 
                        'Constraint_type': 'lt'}
        
        
        self.Fire_use = {'crb': {'bool': {'constant':1}, 
                                 'ba': 'tree_mod', 
                                 'size': 0.666666667}}
        
        self.Defor_size = 1.8
        
        self.Nfer_use = {'tree': 'tree_mod'}
        
        
class MOSH(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Cropland'
        self.Habitat = {'Map': 'HDI', 
                        'Constraint': 0.84, 
                        'Constraint_type': 'lt'}
        
        
        self.Fire_use = {'crb': {'bool': 'tree_mod', 
                                 'ba': 'tree_mod', 
                                 'size': 0.9920635}}
        
        self.Defor_size = 4.507162
        
        self.Nfer_use = {'tree': 'tree_mod'}
        
        
    def fire_use(self):
    
        fire_dat = {}
        self.Fire_vals= {}

        probs_key     = {'bool': 'yval', 
                         'ba'  : 'yval'} ### used for gathering final results
        
        
        ### gather numpy arrays 4 predictor variables
        for x in self.Fire_use.keys():
            
            fire_dat[x]        = get_LU_dat(self, probs_key, self.Fire_vars[x])
            self.Fire_vals[x]  = predict_LU_behaviour(self, probs_key, 
                                 self.Fire_vars[x], fire_dat[x], 
                                 self.Fire_use[x], normalise = False)
            
            ### calculate burned area through bool & ba%
            self.Fire_vals[x] = np.nanmin(np.array([self.Fire_vals[x]['bool'], 
                                          self.Fire_vals[x]['ba']]), axis= 0)
            
        ### Adjust for AFT specific constraints
        self.fire_constraints()
        

class Intense_arable(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Cropland'
        self.Habitat = {'Map': 'Pop', 
                        'Constraint': 900, 
                        'Constraint_type': 'lt'}
        
        
        self.Fire_use = {'crb': {'bool': 'tree_mod', 
                                 'ba': {'constant':0.0425}, 
                                 'size': 33.75}}
        
        self.Defor_size = 81.95727
        
        self.Nfer_use = {'tree': 'tree_mod'
                         'lm'  : 'lm'}

