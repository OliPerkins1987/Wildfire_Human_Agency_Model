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


class Agroforestry(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Forestry'
        self.sub_AFT = {'exists': True, 'kind': 'Fraction',
                         'afr': 'Trans', 'ls': 'Forestry'} 
        

class Logger(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Forestry'
        self.sub_AFT = {'exists': True, 'kind': 'Fraction',
                         'afr': 'Trans', 'ls': 'Forestry'} 
        
        
class Managed_forestry(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Forestry'
        self.sub_AFT = {'exists': True, 'kind': 'Addition', 
                        'afr': 'Trans', 'ls': 'Forestry'} 


class Abandoned_forestry(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Post'
        self.ls  = 'Forestry'
        self.sub_AFT = {'exists': False} 


