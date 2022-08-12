# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.AFTs.agent_class import AFT

###########################################################################################

### Nonex AFTs

###########################################################################################


class Hunter_gatherer(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Nonex'
        self.sub_AFT = {'exists': False}    

        self.Fire_use = {'hg': {'bool': 'tree_mod', 
                                 'ba': 'lin_mod', 
                                 'size': 1.27}, 
                         'pyrome': {'bool': 'tree_mod', 
                                    'ba': 'tree_mod', 
                                    'size': 1}}

    def fire_constraints(self):
        
        ### constrain fire use by forest cover
        Forest_mask              = 1 - (self.model.p.Maps['Primary_forest'][self.model.timestep, :, :].data + 
                                         (0.5 * self.model.p.Maps['Secondary_forest'][self.model.timestep, :, :].data))
        self.Fire_vals['hg']     = self.Fire_vals['hg'] * Forest_mask.reshape(self.model.p.xlen * self.model.p.ylen)
        self.Fire_vals['pyrome'] = self.Fire_vals['pyrome'] * Forest_mask.reshape(self.model.p.xlen * self.model.p.ylen)
        
        ### constrain hg for proximity to wealthy cities
        threshold            = self.model.p.Constraint_pars['HG_Market_constraint']
        MI_constraint        = self.model.p.Maps['Market_influence'][self.model.timestep, :, :].data
        self.Fire_vals['hg'] = self.Fire_vals['hg'] * (MI_constraint.reshape(self.model.p.xlen * self.model.p.ylen) < threshold)


class Recreationalist(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Nonex'
        self.sub_AFT = {'exists': True, 'kind': 'Multiple', 
                        'afr': ['Trans', 'Post'], 'ls': ['Nonex', 'Nonex']}



class SLM(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Nonex'
        self.sub_AFT = {'exists': True, 'kind': 'Addition', 
                        'afr': 'Post', 'ls': 'Nonex'} 

        self.Fire_use = {'pyrome': {'bool': 'tree_mod', 
                                    'ba'  : 'tree_mod', 
                                    'size': 4.75}}


class Conservationist(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Post'
        self.ls  = 'Nonex'
        self.sub_AFT = {'exists': True, 'kind': 'Multiple',
                         'afr': ['Trans', 'Post'], 
                         'ls': ['Nonex', 'Nonex']} 
        
        self.Fire_use = {'pyrome': {'bool': 'tree_mod', 
                                    'ba'  : 'tree_mod', 
                                    'size': 150}}

    def fire_constraints(self):
              
        
        ### constrain fire use by forest cover
        Forest_mask              = 1 - (self.model.p.Maps['Primary_forest'][self.model.timestep, :, :].data + 
                                         (0.5 * self.model.p.Maps['Secondary_forest'][self.model.timestep, :, :].data))
        
        self.Fire_vals['pyrome'] = self.Fire_vals['pyrome'] * Forest_mask.reshape(self.model.p.xlen * self.model.p.ylen)


