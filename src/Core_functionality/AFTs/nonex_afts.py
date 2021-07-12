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
        self.sub_AFT = {'exists': True, 'kind': 'Multiple', 
                        'afr': ['Intense', 'Post'], 'ls': ['Nonex', 'Nonex']} 


class Conservationist(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Post'
        self.ls  = 'Nonex'
        self.sub_AFT = {'exists': True, 'kind': 'Multiple',
                         'afr': ['Trans', 'Intense', 'Post'], 
                         'ls': ['Nonex', 'Nonex', 'Nonex']} 
        
        



