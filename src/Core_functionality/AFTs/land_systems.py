# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap

from copy import deepcopy

###########################################################################################

### Prescribed input LS

###########################################################################################

class Cropland(land_system):

    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Prescribed'


class Pasture(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Prescribed'
        
        
class Rangeland(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Prescribed'


class Urban(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Prescribed'



#################################################################################

### LS distributed through competition 

#################################################################################

class Unoccupied(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Competition'
        self.pars_key    = 'Xaxis/Unoccupied'

class Forestry(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Competition'
        self.pars_key    = 'Xaxis/Forestry'

class Nonex(land_system):
    
    ### Takes the remainder after allocation of Unoccupied for 'other' & 
    ### Forestry & Unoccupied for forest
    ### Allocation is in the WHAM! object
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Specified'
    
    def get_vals(self):
        
        self.Dist_vals = np.zeros(self.model.xlen * self.model.ylen)
    
