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
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.prediction_tools.predict_funs import get_LU_dat, predict_LU_behaviour


class fire_control_measures(AFT):
    
    ''' model distribution of fire control behaviour as a function of AFR'''
    
    def setup(self):
        
        self.control_pars = self.model.p.AFT_pars['Fire_escape']['fire_types']
        
        
    def get_afr(self):
    
        tmp_dict = deepcopy(self.model.AFR)
        tmp_dict.update((x, np.array(y).reshape(self.model.ylen *  
                                   self.model.xlen)) for x, y in tmp_dict.items())
        
        self.Control_dat = pd.DataFrame(tmp_dict)
        
        ### make column for max fire regime
        self.Control_dat['Regime_max'] = np.argmax(self.Control_dat.values,axis=1)
        self.Control_dat['Regime_max'] = np.select([self.Control_dat['Regime_max'] % 2 == 0], 
                                                   [1], default = 0)   
    
    def control(self):
        
        self.get_afr()
        self.Control_vals    = {}
        n_col                = self.Control_dat.shape[1]
        
        for f in self.control_pars.keys():
        
            Control_struct       = define_tree_links(self.control_pars[f])

            self.Control_vals[f] = predict_from_tree_fast(dat =  self.Control_dat, 
                                    tree = self.control_pars[f], struct = Control_struct, 
                                     prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0)
        
            self.Control_dat     = self.Control_dat.iloc[:, 0:(n_col+1)]
        
       
        
       
        
       