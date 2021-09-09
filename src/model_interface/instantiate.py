# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:38:02 2021

@author: Oli
"""

#### Load
from model_interface.wham import WHAM
from Core_functionality.AFTs.agent_class import AFT

from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p
from Core_functionality.AFTs.forestry_afts  import Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts  import Hunter_gatherer, Recreationalist, SLM, Conservationist

from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

from Core_functionality.top_down_processes.background_ignitions import background_rate
from Core_functionality.top_down_processes.background_ignitions import arson

#################################################

### Instantiate

#################################################

all_afts = [Swidden, SOSH, MOSH, Intense_arable, 
            Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p,
            Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer, Recreationalist, SLM, Conservationist]

parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': all_afts,
    'LS'  : [Cropland, Rangeland, Pasture, Forestry, Urban, Nonex, Unoccupied],
    'Fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation'}, 
    'Observers': {'background_rate': background_rate, 
                  'arson': arson},
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'end_run' : 0,
    'reporters': ['Managed_fire', 'Background_ignitions', 'Arson'],
    'theta'    : 0.1,
    'bootstrap': False
    
    }

test = WHAM(parameters)

### setup
test.setup()

### go
test.go()



