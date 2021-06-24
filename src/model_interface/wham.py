# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.aft_setup.afts import Swidden, SOSH, MOSH, Intense_arable

class WHAM(ap.Model):

    def setup(self):

        # Parameters
        self.xlen = self.p.xlen
        self.ylen = self.p.ylen

        # Create grid
        self.grid = ap.Grid(self, (self.xlen, self.ylen), track_empty=False)
        
        # Create AFTs
        self.agents = ap.AgentList(self, 
                       [y[0] for y in [ap.AgentList(self, 1, x) for x in self.p.AFTs]])

    
    ##############################
    
    ### AFT distribution function
    
    ###############################
    
    def allocate_X_axis(self):
        
        pass
    
    
    def allocate_space(self):
        
        ### Y-axis
        
        land_systems = pd.Series([x for x in self.agents.ls]).unique()[0]
        afrs         = pd.Series([x for x in self.agents.afr]).unique()[0]
        ls_scores    = {}
    
        for l in land_systems:
            
            ls_scores[l] = [x.Dist_vals for x in self.agents if x.ls == l]
            
        ### X-axis * Y-axis
    

    ### scheduler, recorder, end conditions
    def update(self):
        
        pass
    
    
    def step(self):
        
        ### AFT distribution
        self.agents.compete()
        self.allocate_space()
        
        ### Fire
        
    def end(self):
        
        pass
    
    
    
#################################################

### Instantiate

#################################################

parameters = {
    
    'xlen': 100, 
    'ylen': 100,
    'AFTs': [Swidden, SOSH, MOSH, Intense_arable],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'Theta'    : 0.1
    
    }

test = WHAM(parameters)

test.setup()
test.agents.setup()
test.agents.get_pars(test.p.AFT_pars)
test.agents.compete()
