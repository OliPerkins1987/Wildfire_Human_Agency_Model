# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:25:15 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
exec(open("test_setup.py").read())

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree
from Core_functionality.AFTs.agent_class import AFT, dummy_agent
from Core_functionality.AFTs.afts import SOSH, Intense_arable
from model_interface.wham import WHAM


#########################################################################

### Load test data

#########################################################################

os.chdir(str(test_dat_path) + '\AFTs')
Dummy_frame       = pd.read_csv('Dummy_pars.csv')
Dummy_AFT_frame   = pd.read_csv('Dummy_AFT_pars.csv')
Dummy_dat     = nc.Dataset('Test.nc')
Dummy_dat     = Dummy_dat['Forest_frac'][:]
Map_data      = {'Test': Dummy_dat}
Map_test      = np.array(pd.read_csv('Test_raster.csv'))

### Mock load up
Core_pars = {'AFT_dist': {}, 
             'Fire_use': {}} 

Core_pars['AFT_dist']['Test/Test']          = Dummy_frame
Core_pars['AFT_dist']['Sub_AFTs/Test_Test'] = Dummy_AFT_frame

### Mock model
parameters = {
    
    'xlen': 144, 
    'ylen': 192,
    'AFTs': [dummy_agent, dummy_agent],
    'LS'  : [],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1
    
    }


##########################################################################

### tests

##########################################################################

def test_AFT_dat_load():
    
    mod = WHAM(parameters)
    mod.setup()
    mod.agents.setup()
    mod.agents.get_pars(mod.p.AFT_pars)
    
    assert(np.array_equal(mod.agents.AFT_frame[0].dummy_agent, Dummy_AFT_frame.dummy_agent))


def test_AFT_compete():
    
    errors = []
    
    mod = WHAM(parameters)
    mod.setup()
    mod.agents.setup()
    mod.agents.get_pars(mod.p.AFT_pars)
    mod.agents.compete()
    mod.agents.sub_compete()
                
    if not np.array_equal(np.array(mod.agents.AFT_dat[0]).reshape(144, 192), Dummy_dat):
        errors.append("Data manipulation error in AFT prediction")
            
    if not np.array_equal(mod.agents[0].AFT_vals.value_counts().values, np.array([27233, 415])):
        errors.append("AFT Prediction error")

    if not mod.agents[0].AFT_vals[0] == 0.22327044:
        errors.append("Incorrect ordering or manipulation of raster values")
        
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    


