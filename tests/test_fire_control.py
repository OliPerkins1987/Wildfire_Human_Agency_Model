# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:25:36 2021

@author: Oli
"""


import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

os.chdir(wd)
exec(open("setup_full.py").read())

### bespoke parameters
parameters['escaped_fire'] = False
parameters['reporters']    = []
parameters['write_annual'] = False

### setup mod
mod = WHAM(parameters)
mod.setup()
mod.go()
mod.Observers['fire_control_measures'][0].control()

def test_control_fundamentals():
    
    errors = []
    summary = pd.Series(mod.Observers['fire_control_measures'][0].Control_vals['crb']).value_counts()
    summary2= pd.Series(mod.Observers['fire_control_measures'][0].Control_vals['pasture']).value_counts()
    
    if not summary.index[0] == 1.0:
        
        errors.append("Error in fire control measure predictions")
    
    if not np.max(summary.index) == 1.0 and np.median(summary.index) == 0.865902255:
        
        errors.append("Error in cropland fire control measure predictions")
    
    if not summary2.iloc[2] == 1127:
        
        errors.append("Error in pasture fire control measure predictions")
       
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
    