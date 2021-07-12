# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:38:02 2021

@author: Oli
"""

'''

### This set of test should be used to check macro-scale model outputs 
###  against internal logic and R-based calculations 

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.AFTs.afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex
from model_interface.wham import WHAM

os.chdir(os.path.dirname(os.path.realpath(__file__)))
exec(open("test_setup.py").read())
exec(open(real_dat_path + "\load_up.py").read())


os.chdir(str(test_dat_path) + '\R_outputs')
X_axis = pd.read_csv('X_axis_1990.csv')
JULES_mask = pd.read_csv('JULES_mask.csv')

#################################################

### Instantiate

#################################################

parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [Swidden, SOSH, MOSH, Intense_arable],
    'LS'  : [Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1
    
    }

test = WHAM(parameters)

### setup
test.setup()
test.ls.setup()
test.ls.get_pars(test.p.AFT_pars)
test.agents.setup()
test.agents.get_pars(test.p.AFT_pars)

### ls
test.ls.get_vals()
test.allocate_X_axis()

### AFT
test.agents.compete()
test.allocate_Y_axis()
test.agents.sub_compete()
test.allocate_AFT()

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
 "<class 'numpy.ndarray'>",
 "<class 'dict'>"])


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

    
def test_Y_axis_output():
    
    errors = []
    
    Forestry_results = pd.DataFrame(dict(zip(test.LFS['Forestry'].keys(), 
                         [x.reshape(27648) for x in test.LFS['Forestry'].values()])))
    
    Forestry_results['Total'] = Forestry_results.sum(axis= 1)
    Forestry_results['X_axis']= test.X_axis['Forestry'].reshape(27648)
    Forestry_results['Error'] = Forestry_results['Total'] - Forestry_results['X_axis']
    
    if not np.mean(np.abs(Forestry_results['Error'])) < 0.001 and np.max(Forestry_results['Error']):
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
