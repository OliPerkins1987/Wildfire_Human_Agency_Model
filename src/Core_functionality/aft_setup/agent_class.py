# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np


class AFT(ap.Agent):
    
    ''' 
    Core model class containing key drivers of model function
    '''
    
    def setup(self):
        
        '''
        
        Basic constants:
        ls = land system of AFT
        afr= anthropogenic fire regime of AFT
        fc = fractional coverage of AFT's grid cell
            
        '''  
      
        self.ls = ''
        self.afr= ''
        self.fc = 0
        
    def get_pars(self):
        
        pass
        
    def compete(self):
        
        pass
    
    def fire_use(self):
        
        pass
    
    
    def fire_suppression(self):
        
        pass


