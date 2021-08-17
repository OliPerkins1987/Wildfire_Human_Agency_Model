# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:38:02 2021

@author: Oli
"""



### This set of test should be used to check macro-scale model outputs 
###  against internal logic and R-based calculations 
### Due to load times - run as a standalone experiment


#### Load
import pytest
import numpy as np
import pandas as pd
import os

from model_interface.wham import WHAM
from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p
from Core_functionality.AFTs.forestry_afts  import Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts  import Hunter_gatherer, Recreationalist, SLM, Conservationist
from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

#################################################

### Load real data

#################################################

'''

#################################################

### Instantiate

#################################################

parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [Swidden, SOSH, MOSH, Intense_arable, 
             Pastoralist, Ext_LF_r, Int_LF_r, 
             Ext_LF_p, Int_LF_p, 
             Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer, Recreationalist, SLM, Conservationist],
    'LS'  : [Cropland, Rangeland, Pasture, Forestry, Urban, Nonex, Unoccupied],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 12,
    'end_run' : 12,
    'theta'    : 0.1,
    'bootstrap': False
    
    }

test = WHAM(parameters)

### setup
test.setup()

### go
test.go()


##############################################################################

### tests

##############################################################################

def test_X_axis_na():
    
    assert(np.min([np.min(x) for x in test.ls.Dist_vals][0:6]) == 0)
        
    
def test_X_data_format():
    
    assert([str(type(x)) for x in test.ls.Dist_vals] == ["<class 'numpy.ndarray'>",
 "<class 'numpy.ndarray'>",
 "<class 'numpy.ndarray'>",
 "<class 'numpy.ndarray'>",
 "<class 'numpy.ndarray'>",
 "<class 'dict'>",
 "<class 'numpy.ndarray'>" ])


def test_X_axis_total():
    
    X_vals = pd.DataFrame.from_dict(dict(zip(test.X_axis.keys(), 
                [x.reshape(test.p.xlen*test.p.ylen) for x in test.X_axis.values()])))
    
    X_vals['Sum']          = X_vals.sum(axis = 1)
    X_vals['Error']        = X_vals['Sum'] - test.p.Maps['Mask']

    errors = []
    
    if not len(np.where(X_vals['Error'] < -0.01)[0]) < 10:
        errors.append("X_axis not aligned with mask")
    
    if not len(np.where(X_vals['Error'] >  0.01)[0]) < 10:
        errors.append("X_axis not aligned with mask")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

    
def test_Y_axis_Forestry():
    
    errors = []
    
    Forestry_results = pd.DataFrame(dict(zip(test.LFS['Forestry'].keys(), 
                         [x.reshape(27648) for x in test.LFS['Forestry'].values()])))
    
    Forestry_results['Total'] = Forestry_results.sum(axis= 1)
    Forestry_results['X_axis']= test.X_axis['Forestry'].reshape(27648)
    Forestry_results['Error'] = Forestry_results['Total'] - Forestry_results['X_axis']
    
    if np.any(np.abs(Forestry_results['Error'][Forestry_results['Total'] != 0]) > 0.01):
        errors.append("Forestry LFS do not match mask")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_AFT_output():
    
    errors = []
    
    Forestry_results = pd.DataFrame(dict(zip(test.AFT_scores.keys(), 
                         [x.reshape(27648) for x in test.AFT_scores.values()])))
    
    Forestry_results['Total'] = Forestry_results.sum(axis= 1)
    Forestry_results['X_axis']= test.X_axis['Forestry'].reshape(27648)
    Forestry_results['Error'] = Forestry_results['Total'] - Forestry_results['X_axis']
    
    if not np.mean(np.abs(Forestry_results['Error'])) < 0.001 and np.max(Forestry_results['Error']) < 0.001:
        errors.append("Forestry AFTs do not match mask")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

'''
