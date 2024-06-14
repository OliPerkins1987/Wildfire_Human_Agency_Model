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

### Arson class

###########################################################################################


class arson(AFT):
    
    
    def setup(self):
        AFT.setup(self)
        
        
    def ignite(self):
    
    ###############################################################
                
    ### Define plausible space
                
    ###############################################################    
    
        fire_hab = (self.model.p.Maps['ET'].data[self.model.timestep, :, :] > 641)

    ###############################################################
                
    ### Adjust arson ignitions for specific land use conflicts
                
    ###############################################################
    
        afr_vals = []
    
        ### Assume Nonex doesn't commit arson
        for ls in ['Cropland', 'Pasture', 'Rangeland', 'Forestry', 'Nonex']:
        
            if 'Trans' in self.model.LFS[ls].keys():
                
                afr_vals.append(self.model.LFS[ls]['Trans'])
        
        ### remove agroforestry from logging contribution to arson
        if 'Agroforestry' in self.model.AFT_scores.keys():
            
            afr_vals.append(0 - self.model.AFT_scores['Agroforestry'])
                
        afr_vals = np.nansum(afr_vals, axis = 0)
        
        ###############################################################
        ### adjust for protected areas
        ###############################################################
        if 'Recreationalist' in self.model.AFT_scores.keys():
            
            afr_vals = afr_vals * self.model.AFT_scores['Recreationalist']
        
        afr_vals = np.select([afr_vals > 0], [afr_vals], default = 0)
        
        ### Simple representation of land conflict
        self.Fire_vals = (1/(1+np.exp(
            0-(afr_vals*21.8860313 - 1.8968080 -0.0009795 * self.model.p.Maps['Market.Inf'].data[self.model.timestep, :, :])))) 
        
        ### account for higher HDI regions
        self.Fire_vals = self.Fire_vals * (
            1/(1+np.exp(0-(3.999-10.416*self.model.p.Maps['HDI'].data[self.model.timestep, :, :])))) 
        
        ### adjust for land area of pixel & ecological limits
        self.Fire_vals = self.Fire_vals * fire_hab
        self.Fire_vals = self.Fire_vals * self.model.p.Maps['Mask'].reshape(self.model.ylen, self.model.xlen)
        
        
    ################################################################
    
    ### Add constraints
    
    ################################################################
    
        for c in self.model.Observers.values():
            
            if 'ct' in type(c[0]).__name__:
                
                c.constrain_arson()
        
