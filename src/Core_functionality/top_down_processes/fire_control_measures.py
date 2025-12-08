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
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_numpy
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation


class fire_control_measures(AFT):
    
    ''' model distribution of fire control behaviour as a function of AFR'''
    
    def setup(self):
        
        self.control_pars = self.model.p.AFT_pars['Fire_escape']['fire_types']
        
        
    def get_afr(self):
    
        afr_res = {}
    
        for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
            afr_vals = []
    
            for ls in ['Cropland', 'Rangeland', 'Pasture', 'Forestry', 'Nonex']:
        
                if afr in self.model.LFS[ls].keys():
                
                    afr_vals.append(self.model.LFS[ls][afr])
               
            afr_res[afr] = np.sum(afr_vals, axis = 0).reshape(self.model.p.xlen*self.model.p.ylen)
    
        self.Control_dat = np.array([x for x in afr_res.values()]).transpose()
            
    
    def control(self):
        
        self.get_afr()
        self.Control_vals    = {}
        n_col                = self.Control_dat.shape[1]
        
        for f in self.control_pars.keys():
        
            Control_struct       = define_tree_links(self.control_pars[f])

            self.Control_vals[f] = predict_from_tree_numpy(dat =  self.Control_dat, 
                                    tree = self.control_pars[f], split_vars = ['Pre', 'Trans', 'Intense', 'Post'], 
                                     struct = Control_struct, prob = 'yprob.TRUE', 
                                     skip_val = -1e+10, na_return = 0)
        
            self.Control_dat     = self.Control_dat[:, 0:(n_col+1)]
        
       
        
       
        
       