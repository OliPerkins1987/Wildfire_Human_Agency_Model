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


class deforestation(AFT):
    
    
    def setup(self):
        
        self.VC_rate = {'Cropland':self.model.p.Maps['Arable_deforestation'].data, 
                        'Pasture': self.model.p.Maps['Livestock_deforestation'].data}
        

    def clear_vegetation(self):
        
        ### get AFT data - relevant to defor
        df_a = [hasattr(x, 'Defor_size') for x in self.model.agents]
        a    = [type(x).__name__ for x in self.model.agents]       
        df_a = [i for (i, v) in zip(a, df_a) if v]
        
        ### fire sizes
        Sizes = dict(zip(df_a, 
                         [self.model.agents[a.index(x)].Defor_size/100 for x in df_a]))
        
        ### Distribution
        AFTs  = dict(zip(df_a, 
                         [self.model.AFT_scores[x] for x in df_a]))
        
       
        ### rate of defor
        self.LC_area         = {'Cropland': self.model.X_axis['Cropland'], 
                                'Pasture' : self.model.X_axis['Pasture']}
        
        self.VC_vals         = {'Cropland': self.VC_rate['Cropland'][self.model.timestep, :, :], 
                                'Pasture': self.VC_rate['Pasture'][self.model.timestep, :, :]}

    
        ### calculate vegetation clearance fire
        ### could this be done more neatly?
        
        self.VC_vals  = {'Cropland' :[(self.model.p.Defor_pars['Pre'] * self.VC_vals['Cropland'] * (AFTs['Swidden']/self.LC_area['Cropland'])),
                          (self.model.p.Defor_pars['Trans'] *self.VC_vals['Cropland'] * (AFTs['SOSH']/self.LC_area['Cropland'])),
                          (self.model.p.Defor_pars['Trans'] *self.VC_vals['Cropland'] * (AFTs['MOSH']/self.LC_area['Cropland'])),
                           (self.model.p.Defor_pars['Intense'] *self.VC_vals['Cropland'] * (AFTs['Intense_arable']/self.LC_area['Cropland']))],
                         
                         'Pasture'  :[(self.model.p.Defor_pars['Trans'] *self.VC_vals['Pasture'] * (AFTs['Ext_LF_p']/self.LC_area['Pasture'])),
                           (self.model.p.Defor_pars['Intense'] *self.VC_vals['Pasture'] * (AFTs['Int_LF_p']/self.LC_area['Pasture']))]}
       
        
        ### Compile and calculate ignitions
        self.VC_igs   = [x for y in self.VC_vals.values() for x in y]
        self.VC_igs   = [x / y for x, y in zip(self.VC_igs, Sizes.values())]
        self.VC_igs   = np.nansum(self.VC_igs, axis = 0)
        
        self.VC_vals  = np.nansum([np.nansum(self.VC_vals['Cropland'], axis = 0), 
                                  np.nansum(self.VC_vals['Pasture'], axis = 0)], axis = 0)
        
        
        
       
        
       