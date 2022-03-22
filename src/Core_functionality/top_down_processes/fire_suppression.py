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
        self.Sup_pars['convert'] = [0,0.1, 0.7, 0.95] if self.model.p.Suppression_rate == 'Default' else self.model.p.Suppression_rate
        
        self.Structure           = {'bool': define_tree_links(self.Sup_pars['pred']['bool'])}
        
        
    def suppress(self):
        
       self.Sup_vals= {'bool': '', 
                       'ba'  : ''}      
       
       probs_key    = {'bool': ['Pre', 'Trans', 'Intense', 'Post'], 
                         'ba'  : 'yval'}
       
       ### get data
       self.Sup_dat = dict(zip([x for x in self.model.AFR.keys()], 
                      [pd.Series(x.reshape(self.model.p.xlen * self.model.p.ylen)) for x in self.model.AFR.values()]))
       
       self.Sup_dat = pd.DataFrame(self.Sup_dat)
       
       ###########################
       ### tree mod
       ###########################
       
       self.Sup_vals['bool'] = {}
       
       for i in probs_key['bool']:
       
           self.Sup_vals['bool'][i] = predict_from_tree_fast(dat = deepcopy(self.Sup_dat), 
                                tree = self.Sup_pars['pred']['bool'], struct = self.Structure['bool'], 
                                 prob = i, skip_val = -3.3999999521443642e+38, 
                                 na_return = 0)
       
       ### multiply by suppression factors
       self.Sup_vals['bool'] = [x * y for x, y in zip(self.Sup_vals['bool'].values(), self.Sup_pars['convert'])]
       self.Sup_vals['bool'] = np.nansum(self.Sup_vals['bool'], axis = 0)
         
       ###########################
       ### lin mod
       ###########################
       
       self.Sup_vals['ba']  = deepcopy(self.Sup_dat)
                    
       ### Mulitply data by regression coefs
       for coef in self.Sup_pars['pred']['ba']['var'][1:]:
                        
          self.Sup_vals['ba'][coef] = self.Sup_vals['ba'][coef] * self.Sup_pars['pred']['ba']['coef'].iloc[np.where(self.Sup_pars['pred']['ba']['var'] == coef)[0][0]]
          
       ### Add intercept
       self.Sup_vals['ba'] = self.Sup_vals['ba'].sum(axis = 1) + self.Sup_pars['pred']['ba']['coef'].iloc[np.where(self.Sup_pars['pred']['ba']['var'] == 'Intercept')[0][0]]
                    
       ### Combine
       self.Sup_vals = np.nanmean([x for x in self.Sup_vals.values()], axis = 0)
       
       ### Multiply by Occupied fraction
       self.Sup_vals = self.Sup_vals.reshape(self.model.p.ylen, self.model.p.xlen) * (1-self.model.X_axis['Unoccupied'])
       

