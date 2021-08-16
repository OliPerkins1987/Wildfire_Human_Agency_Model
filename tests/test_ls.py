# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:25:15 2021

@author: Oli
"""

import pytest 
import agentpy as ap
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
Forest_nonex  = pd.read_csv('Nonex_Forest.csv')
Other_nonex   = pd.read_csv('Nonex_Other.csv')

### Mock load up
Core_pars = {'AFT_dist': {'Xaxis/Unoccupied': Nonex_frame, 
                          'Xaxis/Forest': Forest_nonex, 
                          'Xaxis/Other': Other_nonex}, 
             'Fire_use': {}} 

Map_data                  = {}
Map_data['Cropland']      = Dummy_dat
Map_data['Market_access'] = Dummy_dat
Map_data['HDI']           = Dummy_dat
Map_data['NPP']           = Dummy_dat
Map_data['DEM']           = Dummy_dat
Map_data['HDI_GDP']       = Dummy_dat

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
    'theta'    : 0.1, 
    'bootstrap': False
    
    }
    
    mod = WHAM(parameters)
    mod.setup()

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
    'bootstrap': False
    
    }
    
    errors = []
        
    mod = WHAM(parameters)
    # Parameters
    mod.xlen = mod.p.xlen
    mod.ylen = mod.p.ylen

    # Create grid
    mod.grid = ap.Grid(mod, (mod.xlen, mod.ylen), track_empty=False)
        
        
    # Create land systems
    mod.ls     = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.LS]])
        
    # Create AFTs
    mod.agents = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.AFTs]])


    ### Call land system & AFT set up
    mod.ls.setup()
    mod.ls.get_pars(mod.p.AFT_pars)
    mod.ls.get_vals()


    if not len([x for x in mod.ls.Dist_vals[0] if x == 0]) == 21476:
        errors.append("LS competition test 1 failed")
    
    if not len([x for x in mod.ls.Dist_vals[0] if x == 0.18451178]) == 474:
        errors.append("LS competition test 2 failed")
    
    if not np.max(mod.ls.Dist_vals[0]) == 0.675:
        errors.append("LS competition test 3 failed")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

    
def test_ls_specified():
    
    ### Mock model
    parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [],
    'LS'  : [Nonex],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1, 
    'bootstrap': False
    
    }
    
    errors = []
        
    mod = WHAM(parameters)
    # Parameters
    mod.xlen = mod.p.xlen
    mod.ylen = mod.p.ylen

    # Create grid
    mod.grid = ap.Grid(mod, (mod.xlen, mod.ylen), track_empty=False)
        
        
    # Create land systems
    mod.ls     = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.LS]])
        
    # Create AFTs
    mod.agents = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.AFTs]])


    ### Call land system set up
    mod.ls.setup()
    mod.ls.get_pars(mod.p.AFT_pars)
    mod.ls.get_vals()
    
    ### tests
    if not len([x for x in mod.ls.Dist_vals[0]['Forest'] if x == 0]) == 19092:
        errors.append("LS competition test 1 failed")

    if not len([x for x in mod.ls.Dist_vals[0]['Other'] if x == 0]) == 19092:
        errors.append("LS competition test 2 failed")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

    