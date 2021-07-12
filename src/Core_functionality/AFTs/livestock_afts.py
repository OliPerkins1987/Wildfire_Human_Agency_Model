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

### Livestock AFTs - Rangeland

###########################################################################################

class Pastoralist(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Rangeland'
        self.sub_AFT = {'exists': False}    

class Ext_LF_r(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Rangeland'
        self.sub_AFT = {'exists': False} 
        
        
class Int_LF_r(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Rangeland'
        self.sub_AFT = {'exists': False} 

###########################################################################################

### Livestock AFTs - Pasture

###########################################################################################

class Ext_LF_p(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Pasture'
        self.sub_AFT = {'exists': False} 
        
        
class Int_LF_p(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Pasture'
        self.sub_AFT = {'exists': False} 