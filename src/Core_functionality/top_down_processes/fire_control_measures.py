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
from Core_functionality.prediction_tools.predict_funs import get_LU_dat, predict_LU_behaviour


class fire_control_measures(AFT):
    
    ''' model distribution of fire control behaviour as a function of AFR'''
    
    def setup(self):
        
        self.control_pars = self.model.p.AFT_pars['Fire_escape']['fire_types']
        
        
    def get_afr(self):
    
        tmp_dict             = deepcopy(self.model.AFR)
        tmp_dict['Unoc_pre'] = tmp_dict['Pre'] + self.model.X_axis['Unoccupied']
        tmp_dict.update((x, np.array(y).reshape(self.model.ylen *  
                                   self.model.xlen)) for x, y in tmp_dict.items())
        
        self.Control_dat = tmp_dict

    
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
            
            #self.Control_vals[f]  = (self.Control_vals[f] + 0.5)
        
       
        
       
        
       