# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:16:40 2021

@author: Oli
"""

import sharepy
import io
import pandas as pd
import numpy as np
import netCDF4 as nc
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from copy import deepcopy


###########################################################################

### Map calc
from data_import.api.Access_sharepoint import read_shpt_data, shpt_file_list, write_shpt_data

env_path = Path() / ".env"  # move up one directory
load_dotenv(dotenv_path=env_path)
print(os.getenv("SERVER"))

##########################################################################

### import data

##########################################################################

file_list = shpt_file_list()
matching  = [s for s in file_list if "wham_files" in s]

Core_pars = {'AFT_dist': '', 
             'Fire_use': '',
             'Dist_pars': {'Thresholds': '', 
             'Probabilities': '', 
             'Weighted_thresholds':'',
             'Weighted_probabilities': ''}} #empty dict to house files

##########################################################################

### Get AFT distribution parameters

##########################################################################

### Tree structures

AFT_dist              = [s for s in matching if "AFT Distribution/Trees" in s]
Core_pars['AFT_dist'] = [s for s in AFT_dist if "Tree_frame.csv" in s]

Core_pars_keys        = [x[51:-15] for x in Core_pars['AFT_dist']]
Core_pars_vals        = [read_shpt_data(x) for x in Core_pars['AFT_dist']]
Core_pars['AFT_dist'] = dict(zip(Core_pars_keys, Core_pars_vals))

### Thresholds
Core_pars['Dist_pars']['Thresholds']           = [s for s in AFT_dist if "Thresholds" in s]
Core_pars['Dist_pars']['Weighted_thresholds']  = [s for s in AFT_dist if "Weighted_thresholds" in s]

Core_pars_keys                       = [x[51:-17] for x in Core_pars['Dist_pars']['Thresholds']]
Core_pars_vals                       = [read_shpt_data(x) for x in Core_pars['Dist_pars']['Thresholds']]
Core_pars['Dist_pars']['Thresholds'] = {}

for i in range(len(Core_pars_keys)):
    
    Core_pars['Dist_pars']['Thresholds'].setdefault(Core_pars_keys[i],[]).append(Core_pars_vals[i])

Core_pars_keys                                 = [x[51:-26] for x in Core_pars['Dist_pars']['Weighted_thresholds']]
Core_pars_vals                                 = [read_shpt_data(x) for x in Core_pars['Dist_pars']['Weighted_thresholds']]
Core_pars['Dist_pars']['Weighted_thresholds']  = {}

for i in range(len(Core_pars_keys)):
    
    Core_pars['Dist_pars']['Weighted_thresholds'].setdefault(Core_pars_keys[i],[]).append(Core_pars_vals[i])


### Probs
Core_pars['Dist_pars']['Probs']           = [s for s in AFT_dist if "Probs" in s]
Core_pars['Dist_pars']['Weighted_probs']  = [s for s in AFT_dist if "Weighted_probs" in s]

Core_pars_keys                            = [x[51:-12] for x in Core_pars['Dist_pars']['Probs']]
Core_pars_vals                            = [read_shpt_data(x) for x in Core_pars['Dist_pars']['Probs']]
Core_pars['Dist_pars']['Probs']           = {}

for i in range(len(Core_pars_keys)):
    
    Core_pars['Dist_pars']['Probs'].setdefault(Core_pars_keys[i],[]).append(Core_pars_vals[i])

Core_pars_keys                            = [x[51:-21] for x in Core_pars['Dist_pars']['Weighted_probs']]
Core_pars_vals                            = [read_shpt_data(x) for x in Core_pars['Dist_pars']['Weighted_probs']]
Core_pars['Dist_pars']['Weighted_probs']  = {}

for i in range(len(Core_pars_keys)):
    
    Core_pars['Dist_pars']['Weighted_probs'].setdefault(Core_pars_keys[i],[]).append(Core_pars_vals[i])



###########################################################################

### Get maps

###########################################################################

Mask                 = [s for s in matching if 'JULES_mask' in s]
Maps                 = [s for s in matching if "Dynamic/Maps" in s]
Maps                 = [s for s in matching if ".nc" in s]
dest_folder = r'C:/Users/Oli/Documents/PhD/Model development/Data/wham_dynamic/'
[read_shpt_data(x, download_dir = dest_folder) for x in Maps]

Map_data = dict(zip([x[34:-3] for x in Maps], 
            [nc.Dataset(dest_folder + x[34:-3] + '.nc') for x in Maps]))

var_key  = zip([x for x in Map_data.values()], 
               [[x for x in y.variables.keys()][len(y.variables.keys()) -1 ] for y in Map_data.values()])

Map_data = dict(zip([x for x in Map_data.keys()], 
            [x[y][:] for x, y in var_key]))

Map_data['Mask'] = np.array(read_shpt_data(Mask[0])).reshape(27648)

###########################################################################

### These need sorting out!!

Map_data['Market_influence'] = Map_data['GDP'] * Map_data['Market_access'][0:26, :, :]
Map_data['Market.influence'] = Map_data['GDP'] * Map_data['Market_access'][0:26, :, :]
Map_data['HDI_GDP']          = np.log(Map_data['GDP'].data) * Map_data['HDI']
Map_data['WFI']              = (1/Map_data['TRI']) * Map_data['GDP']


### sort out missing values in processed data
for i in range(Map_data['HDI_GDP'].shape[0]):
    
    for j in range(Map_data['HDI_GDP'].shape[1]):
        
        for k in range(Map_data['HDI_GDP'].shape[2]):
            
            if np.isnan(Map_data['HDI_GDP'].data[i, j, k]):
                 
                Map_data['HDI_GDP'].data[i, j, k] = -3.3999999521443642e+38
            
            if Map_data['WFI'].data[i, j, k] == 0.0:
            
                Map_data['WFI'].data[i, j, k] = -3.3999999521443642e+38


###########################################################################

### Combined weighted/un-weighted thresholds

###########################################################################

for key in Core_pars['Dist_pars']['Thresholds'].keys():
    
    for j in range(len(Core_pars['Dist_pars']['Thresholds'][key])):
        
        Core_pars['Dist_pars']['Thresholds'][key][j] = pd.concat([Core_pars['Dist_pars']['Thresholds'][key][j] , 
                                                                  Core_pars['Dist_pars']['Weighted_thresholds'][key][j]])
        
    for j in range(len(Core_pars['Dist_pars']['Probs'][key])):

        Core_pars['Dist_pars']['Probs'][key][j] = pd.concat([Core_pars['Dist_pars']['Probs'][key][j] , 
                                                                  Core_pars['Dist_pars']['Weighted_probs'][key][j]])


