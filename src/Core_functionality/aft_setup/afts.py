# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.aft_setup.agent_class import AFT

###########################################################################################

### Arable AFTs

###########################################################################################

class Swidden(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Cropland'
        ## needs a third mathod for multiple fractions across LFS
        self.sub_AFT = {'exists': True, 'kind': 'Addition',  
                        'afr': 'Trans', 'ls': 'Cropland'}    ## also shares a part of transition

class SOSH(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Cropland'
        self.sub_AFT = {'exists': True, 'kind': 'Fraction',
                         'afr': 'Trans', 'ls': 'Cropland'} 
        
        
class MOSH(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Cropland'
        self.sub_AFT = {'exists': True, 'kind': 'Fraction',
                         'afr': 'Trans', 'ls': 'Cropland'} 


class Intense_arable(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Cropland'
        self.sub_AFT = {'exists': False} 


#################################################################################

### Livestock AFTs

#################################################################################
