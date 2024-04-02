# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:15:26 2021

Contains constants for setup of test runs

@author: Oli
"""

import os
import pytest

#### Load
from model_interface.wham import WHAM
from Core_functionality.AFTs.agent_class import AFT

from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p
from Core_functionality.AFTs.forestry_afts  import Hunter_gatherer_f, Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts  import Hunter_gatherer_n, Recreationalist, SLM, Conservationist

from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

from Core_functionality.top_down_processes.arson import arson
from Core_functionality.top_down_processes.background_ignitions import background_rate
from Core_functionality.top_down_processes.fire_constraints import fuel_ct, dominant_afr_ct
from Core_functionality.top_down_processes.fire_control_measures import fire_control_measures
from Core_functionality.top_down_processes.deforestation import deforestation


from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap

from dask.distributed import Client
from copy import deepcopy

###############################################################################################
### load data
###############################################################################################

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

os.chdir((wd[0:-6] + '/src/data_import'))
exec(open("local_load_up.py").read())
    

###############################################################################################
### make parameters dictionary for WHAM! object
###############################################################################################

@pytest.fixture(scope="session")
def mod_pars():
    
    all_afts = [Swidden, SOSH, MOSH, Intense_arable, 
            Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p,
            Hunter_gatherer_f, Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer_n, Recreationalist, SLM, Conservationist]

    parameters = {
    
    ### Model run limits
    'xlen': 1440, 
    'ylen': 720,
    'start_run': 0,
    'end_run' : 0,
    
    ### Agents
    'AFTs': all_afts,
    
    'LS'  : [Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex],
    
    'Observers': {'background_rate': background_rate, 
                  'arson': arson, 
                  'fuel_constraint': fuel_ct, 
                  'dominant_afr_constraint': dominant_afr_ct, 
                  'fire_control_measures': fire_control_measures, 
                  'deforestation': deforestation},    
    
    #'Fire_seasonality': Seasonality,
    
    ### data
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    
    ### Fire parameters
    'fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation', 'defor': 'Vegetation'}, 

    ### constraints
    'Constraint_pars': {'Soil_threshold': 0.1325, 
                        'Dominant_afr_threshold': 0.5, 
                        'Rangeland_stocking_contstraint': True, 
                        'R_s_c_Positive' : False, 
                        'HG_Market_constraint': 7800, 
                        'Arson_threshold': 0.5},
    
    'Defor_pars': {'Pre'    : 1, 
                   'Trans'  : 0.84, 
                   'Intense': 0.31},
    
    ### MODIS emulation
    'emulation'    : False, ##if True add 'Emulated_fire' to reporters
    
    ### fire meta pars
    'Seasonality'  : False, 
    'escaped_fire' : True,
    'theta'        : 0.1,

    ### reporters
    'reporters': ['Managed_fire', 'Background_ignitions', 'Arson', 'Escaped_fire'],
    
    ### switch and parameters for bootstrap version of model
    'bootstrap': True,
    'numb_bootstrap': 10, #either int or 'max' for all available
    'n_cores'  : 3,
	
	'write_annual': False
    
    
    }

    return(parameters)
