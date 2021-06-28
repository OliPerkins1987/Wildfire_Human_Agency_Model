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
    
    ### AFT distribution functions
    
    ###############################
    
    def allocate_X_axis(self):
        
        pass
    
    
    def allocate_Y_axis(self):
        
        ### Gather Y-axis from AFTs
        
        land_systems = pd.Series([x for x in self.agents.ls]).unique()[0]
        afrs         = pd.Series([x for x in self.agents.afr]).unique()[0]
        ls_scores    = {}
    
    
        if type(land_systems) == str:
            land_systems = [land_systems] #catch the case where only 1 ls type
            
        for l in land_systems:
            
            ### get predictions
            ls_scores[l] = [x.Dist_vals for x in self.agents if x.ls == l]
                
            ### remove dupes 
            unique_arr   = [np.array(x) for x in set(map(tuple, ls_scores[l]))]
            
            ### calculate total by land system by cell
            tot_y        = np.add.reduce(unique_arr)
            
            ### divide by total & reshape to world map
            ls_scores[l] = dict(zip(list(dict.fromkeys([x.afr for x in self.agents if x.ls == l])), 
                          [np.array(x / tot_y).reshape(self.p.ylen, self.p.xlen) for x in ls_scores[l]]))
               
        
        ### stash LFS scores as Y-axis
        self.Y_axis = ls_scores
        
        
    def allocate_AFT(self):

        pass
    
    #####################################################################################
    
    ### scheduler, recorder, end conditions

    #####################################################################################

    def update(self):
        
        pass
    
    
    def step(self):
        
        ### AFT distribution
        self.agents.compete()
        self.allocate_X_axis()
        self.allocate_Y_axis()
        
        ### Fire
        
    def end(self):
        
        pass
    
    
    
#################################################

### Instantiate

#################################################

parameters = {
    
    'xlen': 144, 
    'ylen': 192,
    'AFTs': [Swidden, SOSH, MOSH, Intense_arable],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1
    
    }

test = WHAM(parameters)

test.setup()
test.agents.setup()
test.agents.get_pars(test.p.AFT_pars)
#test.agents.compete()
#test.allocate_Y_axis()
test.agents.sub_compete()

