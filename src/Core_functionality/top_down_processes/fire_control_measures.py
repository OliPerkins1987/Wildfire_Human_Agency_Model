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
    
        self.Control_dat = afr_res
            
    
    def control(self):
        
        self.get_afr()
        self.Control_vals    = {}
        
        for f in self.control_pars.keys():
            
            Control_vars         = [x for x in self.control_pars[f].iloc[:,1].tolist() if x != '<leaf>']
            Control_dat          = np.array([self.Control_dat[x] for x in Control_vars]).transpose()
                        
            Control_struct       = define_tree_links(self.control_pars[f])

            self.Control_vals[f] = predict_from_tree_numpy(dat =  Control_dat, tree = self.control_pars[f], 
                                   split_vars = Control_vars, struct = Control_struct, 
                                     prob = 'yprob.TRUE', skip_val = -1.0e+10, na_return = 0.5)
            
            #self.Control_dat     = self.Control_dat[:, 0:(n_col+1)]
        
       
        
       
        
       