# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:05:09 2021

@author: Oli
"""

### I am satisfied this is working, just not sure how best to test?

import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

os.chdir((wd[0:-6] + '/src/data_import'))
exec(open("local_load_up.py").read())

os.chdir(str(wd + '/test_data/R_outputs').replace('\\', '/'))
R_fire = pd.read_csv('Fire_1990.csv')
R_afr  = pd.read_csv('afr_constraint_1990.csv')
R_afr['fire']     = R_fire['value']
R_afr['afr_fire'] = R_afr['fire'] * R_afr['value']


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

### ignite
mod.ls.get_vals()
mod.allocate_X_axis()

### afr distribution
mod.agents.compete()
mod.allocate_Y_axis()

afr_res = {}
    
for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
    afr_vals = []
    
    for ls in ['Nonex']:
        
        if afr in mod.model.LFS[ls].keys():
                
            afr_vals.append(mod.model.LFS[ls][afr])
               
    afr_res[afr] = np.nansum(afr_vals, axis = 0)
            
    ### divide by nonex fraction
    afr_res[afr] = afr_res[afr] / mod.model.X_axis['Nonex']
               
### Zero intensive cases below dominance threshold
afr_res['Intense'] = np.select([afr_res['Intense'] >= 0.4], 
                            [afr_res['Intense']], default = 0)
        
### calculate impact of dominant exclusionary afr
Intense = np.nanargmax([x for x in afr_res.values()], axis = 0)
Intense = ((Intense==2) * (1 - afr_res['Intense'])) + (Intense!=2 * 1)

R_afr['python_afr'] = Intense.reshape(27648)
R_afr['ET']         = Map_data['ET'][0].data.reshape(27648) < float(mod.agents[15].Dist_frame['splits.cutleft'][0][1:])
R_afr['HDI']        = Map_data['HDI'][0].data.reshape(27648)  > float(mod.agents[15].Dist_frame['splits.cutleft'][2][1:])

R_afr               = R_afr[R_afr['fire'].isnull() == False]

#######################################################################################

### tests

#######################################################################################

def test_afr_constraint():
    
    ''' based on the idea that where intense is stronger, 
    the constraint will apply more'''

    a = np.nanmean(R_afr['python_afr'][np.logical_and(R_afr['ET'], R_afr['HDI'])])
    b = np.nanmean(R_afr['python_afr'][np.logical_and(R_afr['ET'] == False, R_afr['HDI'])])
    c = np.nanmean(R_afr['python_afr'][np.logical_and(R_afr['ET'] == False, R_afr['HDI'] == False)])

    assert a < b and c == 1.0
    
    

