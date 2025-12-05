# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:38:02 2021

@author: Oli
"""

from model_interface.wham import WHAM
from Core_functionality.AFTs.agent_class import AFT

from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist_r, Pastoralist_p, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p, Abandoned_LF_r
from Core_functionality.AFTs.forestry_afts import Hunter_gatherer_f, Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts import Hunter_gatherer_n, Recreationalist, SLM, Conservationist

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

############################################################################
# Load data
############################################################################

import os
from copy import deepcopy

# Try to import the loader from the package location; if not available, fall back to proposed_edits cleaned loader
try:
    # Preferred: package provides a `load_local_data` function
    from data_import.local_load_up import load_local_data
    Core_pars, Map_data = load_local_data()

except Exception:
    # Last resort: preserve original behaviour by attempting the old exec pattern
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    wd = os.getcwd().replace('\\', '/')
    os.chdir((wd[0:-16] + '/data_import'))

    exec(open("local_load_up.py").read())
            
finally:
    RuntimeError("Data load failed")


#################################################

### setup parameters

#################################################

all_afts = [Swidden, SOSH, MOSH, Intense_arable, 
            Pastoralist_r, Pastoralist_p, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p, Abandoned_LF_r,
            Hunter_gatherer_f, Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer_n, Recreationalist, SLM, Conservationist]

parameters = {
    'xlen': 1440, 
    'ylen': 720,
    'start_run': 0,
    'end_run' : 25,
    'AFTs': all_afts,
    'LS'  : [Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex],
    'theta'   : 0.1,
    'Observers': {'background_rate': background_rate, 
                  'arson': arson, 
                  'fuel_constraint': fuel_ct, 
                  'dominant_afr_constraint': dominant_afr_ct, 
                  'fire_control_measures': fire_control_measures},    
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'AFT_fire': True,
    'AFT_Nfer': False,
    'Policy': True,
    'Fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation'}, 
    'Constraint_pars': {'Soil_threshold': 0.1325, 
                        'Dominant_afr_threshold': 0.5, 
                        'Rangeland_stocking_contstraint': True, 
                        'R_s_c_Positive' : False, 
                        'HG_Market_constraint': 7800, 
                        'Arson_threshold': 0.5},
    'apply_fire_constraints': True,
    'Defor_pars': {'Pre'    : 1, 
                   'Trans'  : 0.84, 
                   'Intense': 0.31},
    'Seasonality'  : False, 
    'escaped_fire' : False, 
    'reporters': ['Managed_fire', 'AFT_scores'],
    'bootstrap': True,
    'numb_bootstrap': 20,
    'n_cores'  : 2,
    'write_annual': True,
    'write_fp': r'C:/Users/Oli/Documents/PIES/WHAMv2/mod/initial_results'  
}


#####################################################

### Run model

#####################################################

if __name__ == "__main__":
    mod = WHAM(parameters)
    mod.setup()
    mod.go()

