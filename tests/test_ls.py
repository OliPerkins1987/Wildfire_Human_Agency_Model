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
from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

from model_interface.wham import WHAM


#########################################################################

### Load test data

#########################################################################

os.chdir(str(test_dat_path) + '\AFTs')
Dummy_dat     = nc.Dataset('Cropland.nc')
Dummy_dat     = Dummy_dat['Crop_frac']
Nonex_frame   = pd.read_csv('Nonex_pars.csv')     


### Mock load up
Core_pars = {'AFT_dist': {'Xaxis/Unoccupied': Nonex_frame}, 
             'Fire_use': {}} 

Map_data                  = {}
Map_data['Cropland']      = Dummy_dat
Map_data['Market_access'] = Dummy_dat



##########################################################################

### tests

##########################################################################

def test_ls_prescribed():
    
    ### Mock model
    parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [],
    'LS'  : [Cropland],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1
    
    }
    
    mod = WHAM(parameters)
    mod.setup()
    mod.ls.setup()
    mod.ls.get_pars(mod.p.AFT_pars)
    mod.ls.get_vals()
        
    Crop_dat = Dummy_dat[0, :, :].data.reshape(144*192)
    Crop_dat = np.array([x if x >= 0 else 0 for x in Crop_dat ])
    
    assert(np.array_equal(mod.ls.Dist_vals[0], 
                          Crop_dat))


def test_ls_compete():
    
    ### Mock model
    parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [],
    'LS'  : [Unoccupied],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1, 
    
    
    }
    
    errors = []
    
    mod = WHAM(parameters)
    mod.setup()
    mod.ls.setup()
    mod.ls.get_pars(mod.p.AFT_pars)
    mod.ls.get_vals()

    if not len([x for x in mod.ls.Dist_vals[0] if x == 0.06155861]) == 2384:
        errors.append("LS competition test 1 failed")
    
    if not len([x for x in mod.ls.Dist_vals[0] if x == 0.18451178]) == 474:
        errors.append("LS competition test 2 failed")
    
    if not np.max(mod.ls.Dist_vals[0]) == 0.675:
        errors.append("LS competition test 3 failed")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
    
    ### Come back to this when missig values are dealt with
    
def test_ls_specified():
    
    pass
    
    