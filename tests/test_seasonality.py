# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:17:04 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os
from copy import deepcopy

os.chdir(os.path.dirname(os.path.realpath(__file__)))
exec(open("test_setup.py").read())

os.chdir(real_dat_path)
exec(open("local_load_up.py").read())


from model_interface.wham import WHAM
from Core_functionality.AFTs.agent_class import AFT

from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p
from Core_functionality.AFTs.forestry_afts  import Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts  import Hunter_gatherer, Recreationalist, SLM, Conservationist

from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

from Core_functionality.top_down_processes.arson import arson
from Core_functionality.top_down_processes.background_ignitions import background_rate
from Core_functionality.top_down_processes.fire_constraints import fuel_ct, dominant_afr_ct

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation


#####################################################################

### Run model year then reproduce outputs

#####################################################################

### Run model for 1 year

all_afts = [Swidden, SOSH, MOSH, Intense_arable, 
            Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p,
            Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer, Recreationalist, SLM, Conservationist]

parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': all_afts,
    'LS'  : [Cropland, Rangeland, Pasture, Forestry, Nonex, Unoccupied, Urban],
    'Fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation'}, 
    'Observers': {'arson': arson, 'background_rate': background_rate},
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'Constraint_pars': {'Soil_threshold': 0.1325, 
                        'Dominant_afr_threshold': 0.5, 
                        'Rangeland_stocking_contstraint': True, 
                        'R_s_c_Positive' : False, 
                        'HG_Market_constraint': 7800, 
                        'Arson_threshold': 0.5},
    'timestep': 0,
    'end_run' : 0,
    'reporters': ['Managed_fire', 'Background_ignitions','Arson'],
    'theta'    : 0.1,
    'bootstrap': False, 
    'Seasonality': False
    
    }


mod = WHAM(parameters)

### setup
mod.setup()

### go
mod.go()

mod_annual = deepcopy(mod.results['Managed_fire'][0]['Total'])

#######################
### Run model monthly
#######################

all_afts = [Swidden, SOSH, MOSH, Intense_arable, 
            Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p,
            Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer, Recreationalist, SLM, Conservationist]

parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': all_afts,
    'LS'  : [Cropland, Rangeland, Pasture, Forestry, Nonex, Unoccupied, Urban],
    'Fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation'}, 
    'Fire_seasonality': Seasonality,
    'Observers': {'arson': arson, 'background_rate': background_rate},
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'Constraint_pars': {'Soil_threshold': 0.1325, 
                        'Dominant_afr_threshold': 0.5, 
                        'Rangeland_stocking_contstraint': True, 
                        'R_s_c_Positive' : False, 
                        'HG_Market_constraint': 7800, 
                        'Arson_threshold': 0.5},
    'timestep': 0,
    'end_run' : 0,
    'reporters': ['Managed_fire', 'Background_ignitions','Arson'],
    'theta'    : 0.1,
    'bootstrap': False, 
    'Seasonality': True
    
    }


mod = WHAM(parameters)

### setup
mod.setup()

### go
mod.go()

##################################

### tests

##################################

def test_seasonality_mean():
    
    seasonal = np.nansum(mod.results['Managed_fire'][0]['Total'], axis = 0)
    
    assert pytest.approx(np.nanmean(mod_annual)) == np.nanmean(seasonal)


def test_seasonality_quantiles():
    
    seasonal = np.nansum(mod.results['Managed_fire'][0]['Total'], axis = 0)
    quants   = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    
    
    assert pytest.approx(np.nanquantile(mod_annual, quants)) == np.nanquantile(seasonal, quants)
