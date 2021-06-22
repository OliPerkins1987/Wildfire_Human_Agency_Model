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


    def update(self):
        
        pass
    
    def step(self):
        
        pass
        
    def end(self):
        
        pass
    
    
    
#################################################

### Instantiate

#################################################

parameters = {
    
    'xlen': 100, 
    'ylen': 100,
    'AFTs'  : [Swidden, SOSH, MOSH, Intense_arable]
    
    }

test = WHAM(parameters)

