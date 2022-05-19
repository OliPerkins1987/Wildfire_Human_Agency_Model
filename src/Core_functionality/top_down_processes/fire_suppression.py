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
from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation



###########################################################################################

### Arson class

###########################################################################################


class fire_fighter(AFT):
    
    def setup(self):
        
        self.Sup_pars            = {'pred'    : self.model.p['AFT_pars']['Fire_suppression']}
                        
    def suppress(self):
        
       self.Sup_vals= {'ba'  : ''}      
       
       ### get data
       self.Sup_dat = dict(zip([x for x in self.model.AFR.keys()], 
                      [pd.Series(x.reshape(self.model.p.xlen * self.model.p.ylen)) for x in self.model.AFR.values()]))
       
       self.Sup_dat = pd.DataFrame(self.Sup_dat)
       
       ###########################
       ### lin mod
       ###########################
       
       self.Sup_vals['ba']  = deepcopy(self.Sup_dat)
                    
       ### Mulitply data by regression coefs
       for coef in self.Sup_pars['pred']['ba']['var'][1:]:
                        
          self.Sup_vals['ba'][coef] = self.Sup_vals['ba'][coef] * self.Sup_pars['pred']['ba']['coef'].iloc[np.where(self.Sup_pars['pred']['ba']['var'] == coef)[0][0]]
          
       ### Add intercept
       self.Sup_vals['ba'] = self.Sup_vals['ba'].sum(axis = 1) + self.Sup_pars['pred']['ba']['coef'].iloc[np.where(self.Sup_pars['pred']['ba']['var'] == 'Intercept')[0][0]]
       
       ### Restrict to 0-1
       self.Sup_vals = np.select([self.Sup_vals['ba'] > 1, self.Sup_vals['ba'] < 0], [1, 0], default = self.Sup_vals['ba'])
       
       ### Multiply by Occupied fraction
       self.Sup_vals = self.Sup_vals.reshape(self.model.p.ylen, self.model.p.xlen) * (1-self.model.X_axis['Unoccupied'])
       
