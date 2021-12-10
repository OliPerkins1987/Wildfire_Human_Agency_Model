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

### Arable AFTs

###########################################################################################

class Swidden(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Cropland'
        self.sub_AFT = {'exists': True, 'kind': 'Addition',  
                        'afr': 'Trans', 'ls': 'Cropland'}    

        self.Fire_use = {'cfp': {'bool': 'tree_mod', 
                                 'ba':   'tree_mod', 
                                 'size': 0.728455277}}
        
        self.Defor_size = 0.833333333


class SOSH(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Cropland'
        self.sub_AFT = {'exists': True, 'kind': 'Fraction',
                         'afr': 'Trans', 'ls': 'Cropland'} 
        
        self.Fire_use = {'crb': {'bool': {'constant':1}, 
                                 'ba': 'tree_mod', 
                                 'size': 0.666666667}}
        
        self.Defor_size = 1.8
        
        
class MOSH(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Cropland'
        self.sub_AFT = {'exists': True, 'kind': 'Fraction',
                         'afr': 'Trans', 'ls': 'Cropland'} 

        self.Fire_use = {'crb': {'bool': {'constant':1}, 
                                 'ba': 'tree_mod', 
                                 'size': 0.9920635}}
        
        self.Defor_size = 4.507162
        

class Intense_arable(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Cropland'
        self.sub_AFT = {'exists': False} 

        self.Fire_use = {'crb': {'bool': 'tree_mod', 
                                 'ba': {'constant':0.0425}, 
                                 'size': 33.75}}
        
        self.Defor_size = 81.95727


