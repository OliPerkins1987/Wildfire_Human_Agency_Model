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

### Forestry AFTs

###########################################################################################

class Hunter_gatherer_f(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Forestry'
        
        self.Habitat = {'Map': 'MA', 
                        'Constraint': 0.5782851, 
                        'Constraint_type': 'lt'} 
        

class Agroforestry(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Forestry'
        
        self.Habitat = {'Map': 'MA', 
                        'Constraint': 0.0059052876, 
                        'Constraint_type': 'gt'} 

class Logger(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Forestry'
        
        self.Habitat = {'Map': 'MA', 
                        'Constraint': 0.0140444132, 
                        'Constraint_type': 'gt'} 

class Managed_forestry(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Forestry'
        
        self.Habitat = {'Map': 'HDI', 
                        'Constraint': 0.3678384, 
                        'Constraint_type': 'gt'} 

        self.Fire_use = {'pyrome':{'bool': 'tree_mod', 
                                   'ba': {'constant': 0.01}, 
                                   'size': 7.5}}

class Abandoned_forestry(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Post'
        self.ls  = 'Forestry'
        
        self.Habitat = {'Map': 'HDI', 
                        'Constraint': 0.3678384, 
                        'Constraint_type': 'gt'} 




