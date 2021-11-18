# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:04:17 2021

@author: Oli
"""

import agentpy as ap
import numpy as np
import pandas as pd
from copy import deepcopy

from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation


class fire_control_measures(AFT):
    
    ''' model distribution of fire control behaviour as a function of AFR'''
    
    def setup(self):
        
        self.control_pars = self.model.p.AFT_pars['Fire_escape']
        
        
    def get_afr(self):
    
        afr_res = {}
    
        for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
            afr_vals = []
    
            for ls in ['Cropland', 'Rangeland', 'Pasture', 'Forestry', 'Nonex']:
        
                if afr in self.model.LFS[ls].keys():
                
                    afr_vals.append(self.model.LFS[ls][afr])
               
            afr_res[afr] = np.nansum(afr_vals, axis = 0).reshape(self.model.p.xlen*self.model.p.ylen)
    
        self.Control_dat = pd.DataFrame(afr_res)
            
    
    def control(self):
        
        self.get_afr()
        self.Control_vals    = {}
        
        for f in self.control_pars.keys():
        
            Control_struct       = define_tree_links(self.control_pars[f])

            self.Control_vals[f] = predict_from_tree_fast(dat =  self.Control_dat, 
                                    tree = self.control_pars[f], struct = Control_struct, 
                                     prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0)
        
            self.Control_dat     = self.Control_dat.iloc[:, 0:4]
        
        