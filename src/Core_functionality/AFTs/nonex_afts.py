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


class Hunter_gatherer_n(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Nonex'
        self.Habitat = {'Map': 'NPP', 
                        'Constraint': 6.805635, 
                        'Constraint_type': 'gt'}     

        self.Fire_use = {'hg': {'bool': 'tree_mod', 
                                 'ba': 'tree_mod', 
                                 'size': 1.34}}

    def fire_constraints(self):
        
        ### constrain fire use by forest cover
        Forest_mask              = 1 - self.model.p.Maps['Forest'][self.model.timestep, :, :].data
        Forest_mask              = np.select([Forest_mask > 1], [0], Forest_mask)
        self.Fire_vals['hg']     = self.Fire_vals['hg'] * Forest_mask.reshape(self.model.p.xlen * self.model.p.ylen)
        
        threshold            = self.model.p.Constraint_pars['HG_Market_constraint']
        MI_constraint        = self.model.p.Maps['Market.Inf'][self.model.timestep, :, :].data
        self.Fire_vals['hg'] = self.Fire_vals['hg'] * (MI_constraint.reshape(self.model.p.xlen * self.model.p.ylen) < threshold)
        
        
class Recreationalist(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Nonex'
        self.Habitat = {'Map': 'Pop', 
                        'Constraint': 400, 
                        'Constraint_type': 'lt'} 
        
        self.Fire_use = {'pyrome': {'bool': 'tree_mod', 
                                    'ba'  : 'tree_mod', 
                                    'size': 150}}
        
    def fire_constraints(self):
        
        ### constrain ontologically: limited management
        ### based on fire use frequency by AFR in DAFI
        self.Fire_vals['pyrome'] = self.Fire_vals['pyrome'] * 0.285
        
        ### constrain fire use by forest cover
        Forest_mask              = 1 - (self.model.p.Maps['Forest'][self.model.timestep, :, :].data)
        Forest_mask              = np.select([Forest_mask > 1], [0], Forest_mask)
        self.Fire_vals['pyrome'] = self.Fire_vals['pyrome'] * Forest_mask.reshape(self.model.p.xlen * self.model.p.ylen)
        

class SLM(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Nonex'
        self.Habitat = {'Map': 'NPP', 
                        'Constraint': 12.149564, 
                        'Constraint_type': 'gt'} 

        self.Fire_use = {'pyrome': {'bool': 'tree_mod', 
                                    'ba'  : 'tree_mod', 
                                    'size': 4.75}}


class Conservationist(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Post'
        self.ls  = 'Nonex'
        self.Habitat = {'Map': 'NPP', 
                        'Constraint': 9.831165, 
                        'Constraint_type': 'gt'}
        
        
        self.Fire_use = {'pyrome': {'bool': 'tree_mod', 
                                    'ba'  : 'tree_mod', 
                                    'size': 150}}



    def fire_constraints(self):
        
        ### constrain fire use by forest cover
        Forest_mask              = 1 - (self.model.p.Maps['Forest'][self.model.timestep, :, :].data)
        Forest_mask              = np.select([Forest_mask > 1], [0], Forest_mask)
        self.Fire_vals['pyrome'] = self.Fire_vals['pyrome'] * Forest_mask.reshape(self.model.p.xlen * self.model.p.ylen)
        

