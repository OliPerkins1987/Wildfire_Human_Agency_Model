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
from Core_functionality.AFTs.livestock_afts import Pastoralist_r,Pastoralist_p, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p, Abandoned_LF_r
from Core_functionality.AFTs.forestry_afts  import Hunter_gatherer_f, Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts  import Hunter_gatherer_n, Recreationalist, SLM, Conservationist

from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

from Core_functionality.top_down_processes.arson import arson
from Core_functionality.top_down_processes.background_ignitions import background_rate
from Core_functionality.top_down_processes.fire_constraints import fuel_ct, dominant_afr_ct
from Core_functionality.top_down_processes.fire_control_measures import fire_control_measures
from Core_functionality.top_down_processes.deforestation import deforestation


from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast, predict_from_tree_numpy
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap

from dask.distributed import Client
from copy import deepcopy

###############################################################################################
### load data
###############################################################################################

from data_import.local_load_up import load_local_data
Core_pars, Map_data = load_local_data()

###############################################################################################
### make parameters dictionary for WHAM! object
###############################################################################################

@pytest.fixture(scope="session")
def mod_pars():
    
    all_afts = [Swidden, SOSH, MOSH, Intense_arable, 
            Pastoralist_r, Pastoralist_p, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p, Abandoned_LF_r,
            Hunter_gatherer_f, Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer_n, Recreationalist, SLM, Conservationist]

    parameters = {
            
    #################
    
    ### meta pars
    
    #################
    
    ### Spatio-temporal limits
    'xlen': 1440, 
    'ylen': 720,
    'start_run': 0,
    'end_run' : 25,
    
    ### Agents
    'AFTs': all_afts,
    
    'LS'  : [Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex],
    
    ### AFT distribution parameter
    'theta'   : 0.1,
    
    'Observers': {'background_rate': background_rate, 
                  'arson': arson, 
                  #'deforestation': deforestation,
                  'fuel_constraint': fuel_ct, 
                  'dominant_afr_constraint': dominant_afr_ct, 
                  'fire_control_measures': fire_control_measures},    
       
    ### data
    'AFT_pars': Core_pars, ##defined in data load
    'Maps'    : Map_data,  ##defined in data load
    
    ### which AFT aspects are being modelled?
    'AFT_fire': False,
    'AFT_Nfer': False,
    'Policy': True, ## policy
    
    ################################
    
    ### Nitrogen pars
    
    ################################
    
    ################################
    
    ### Fire pars
    
    ################################
    
    ### Fire parameters
    'Fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation'}, 

    ### constraints
    'Constraint_pars': {'Soil_threshold': 0.1325, 
                        'Dominant_afr_threshold': 0.5, 
                        'Rangeland_stocking_contstraint': True, 
                        'R_s_c_Positive' : False, 
                        'HG_Market_constraint': 7800, 
                        'Arson_threshold': 0.5},
    
    ### Deforestation fire fraction
    'Defor_pars': {'Pre'    : 1, 
                   'Trans'  : 0.84, 
                   'Intense': 0.31},
    
    
    ### fire meta pars
    #'Fire_seasonality': Seasonality, ##defined in data load 
    'Seasonality'  : False, 
    'escaped_fire' : False, ##if True add 'Escaped_fire' to reporters
    'apply_fire_constraints' : False,
    
    ### reporters
    'reporters': ['Managed_fire', 'Background_ignitions', 'Arson', 'Escaped_fire'],
    
    ### switch and parameters for bootstrap version of model
    'bootstrap': True,
    'numb_bootstrap': 10, #either int or 'max' for all available
    'n_cores'  : 3,
	
	'write_annual': False
    
    
    }

    return(parameters)
