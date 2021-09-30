# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:12:17 2021

@author: Oli
"""

from dask.distributed import Client
import numpy as np
import pandas as pd
from copy import deepcopy
import time

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation

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


all_afts = [Swidden, SOSH, MOSH, Intense_arable, 
            Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p,
            Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer, Recreationalist, SLM, Conservationist]

parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': all_afts,
    'LS'  : [Cropland, Rangeland, Pasture, Forestry, Nonex, Unoccupied, Urban],
    'Fire_types': {'cfp': 'Arable', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation'}, 
    'Fire_seasonality': Seasonality,
    'Observers': {'background_rate': background_rate, 
                  'arson': arson, 
                  'fuel_constraint': fuel_ct, 
                  'dominant_afr_constraint': dominant_afr_ct},    
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
    'reporters': ['Managed_fire', 'Background_ignitions', 'Arson'],
    'theta'    : 0.1,
    'bootstrap': True,
    'Seasonality': True
    
    }

test = WHAM(parameters)

test.setup()


#######################################################################

### Core prediction code

#######################################################################

test_agent = test.agents[2]

test_agent.Dist_vals = []
            
### gather correct numpy arrays 4 predictor variables
test_agent.Dist_dat  = [test_agent.model.p.Maps[x][test_agent.model.p.timestep, :, :] if len(test_agent.model.p.Maps[x].shape) == 3 else test_agent.model.p.Maps[x] for x in test_agent.Dist_vars]

### combine numpy arrays to single pandas       
test_agent.Dist_dat  = pd.DataFrame.from_dict(dict(zip(test_agent.Dist_vars, 
                              [x.reshape(test_agent.model.p.xlen*test_agent.model.p.ylen).data for x in test_agent.Dist_dat])))
        
a = test_agent

#Dist_vals = Parallel(n_jobs=3)(delayed(boot_pred)(test_agent, i) for i in range(5))

boot_pred = {'df':[], 'ds': deepcopy(a.Dist_struct), 
             'dd': deepcopy(a.Dist_dat)}

#a.boot_Dist_pars['Thresholds'][0].shape[0]
for i in range(100):

    boot_pred['df'].append(deepcopy(update_pars(a.Dist_frame, a.boot_Dist_pars['Thresholds'], 
                                    a.boot_Dist_pars['Probs'], method = 'bootstrapped', 
                                    target = 'yprob.TRUE', source = 'TRUE.', boot_int = i)))
          
    print(a.boot_Dist_pars['Probs'][0]['TRUE.'].iloc[i])
    
    
####################################################################################

### Run parallel 

####################################################################################


client = Client(n_workers=4) 
futures = []

t= time.time()

for i in range(200):
    
    future = client.submit(predict_from_tree_fast, dat = boot_pred['dd'], 
                              tree = boot_pred['df'][i], struct = boot_pred['ds'], 
                               prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0)
    futures.append(future)

results = client.gather(futures)
client.close()
print(time.time() - t)

t       = time.time()
results = []

for i in range(100):
    
    df     = deepcopy(boot_pred['df'][i])
    dd     = deepcopy(boot_pred['dd'])
    results.append(predict_from_tree_fast(dat = dd, 
                              tree = df, struct = boot_pred['ds'], 
                               prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0))
    
print(time.time() - t)