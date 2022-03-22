# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:16:40 2021

@author: Oli
"""


import pandas as pd
import numpy as np
import netCDF4 as nc
import os
import re
from copy import deepcopy


##########################################################################

### import data

##########################################################################

### Set these to your local directories!

root       = r'F:/PhD/Model files/'
Map_folder = r'C:/Users/Oli/Documents/PhD/Model development/Data/wham_dynamic/'

Rlen       = len(root)
Mlen       = len(Map_folder)

file_list = []

for path, subdirs, files in os.walk(root):
    for name in files:
        file_list.append(os.path.join(path, name))

file_list = [s.replace('\\', '/') for s in file_list]

#empty dict to house files
Core_pars = {'AFT_dist': '', 
             'Fire_use': {},
             'Fire_suppression':'',
             'Fire_escape': {},
             'Dist_pars': {'Thresholds': '', 
             'Probabilities': '', 
             'Weighted_thresholds':'',
             'Weighted_probabilities': ''}} 

##########################################################################

### Get AFT distribution parameters

##########################################################################

### Tree structures

AFT_dist              = [s.replace('\\', '/') for s in file_list if "Distribution/Trees" in s]
Core_pars['AFT_dist'] = [s for s in AFT_dist if "Tree_frame.csv" in s]

Core_pars_keys        = [x[(Rlen+19):-15] for x in Core_pars['AFT_dist']]
Core_pars_vals        = [pd.read_csv(x) for x in Core_pars['AFT_dist']]
Core_pars['AFT_dist'] = dict(zip(Core_pars_keys, Core_pars_vals))

### Thresholds
Core_pars['Dist_pars']['Thresholds']           = [s for s in AFT_dist if "Thresholds" in s]
Core_pars['Dist_pars']['Weighted_thresholds']  = [s for s in AFT_dist if "Weighted_thresholds" in s]

Core_pars_keys                       = [x[(Rlen+19):-17] for x in Core_pars['Dist_pars']['Thresholds']]
Core_pars_vals                       = [pd.read_csv(x) for x in Core_pars['Dist_pars']['Thresholds']]
Core_pars['Dist_pars']['Thresholds'] = {}

for i in range(len(Core_pars_keys)):
    
    Core_pars['Dist_pars']['Thresholds'].setdefault(Core_pars_keys[i],[]).append(Core_pars_vals[i])

Core_pars_keys                                 = [x[(Rlen+19):-26] for x in Core_pars['Dist_pars']['Weighted_thresholds']]
Core_pars_vals                                 = [pd.read_csv(x) for x in Core_pars['Dist_pars']['Weighted_thresholds']]
Core_pars['Dist_pars']['Weighted_thresholds']  = {}

for i in range(len(Core_pars_keys)):
    
    Core_pars['Dist_pars']['Weighted_thresholds'].setdefault(Core_pars_keys[i],[]).append(Core_pars_vals[i])


### Probs
Core_pars['Dist_pars']['Probs']           = [s for s in AFT_dist if "Probs" in s]
Core_pars['Dist_pars']['Weighted_probs']  = [s for s in AFT_dist if "Weighted_probs" in s]

Core_pars_keys                            = [x[(Rlen+19):-12] for x in Core_pars['Dist_pars']['Probs']]
Core_pars_vals                            = [pd.read_csv(x) for x in Core_pars['Dist_pars']['Probs']]
Core_pars['Dist_pars']['Probs']           = {}

for i in range(len(Core_pars_keys)):
    
    Core_pars['Dist_pars']['Probs'].setdefault(Core_pars_keys[i],[]).append(Core_pars_vals[i])

Core_pars_keys                            = [x[(Rlen+19):-21] for x in Core_pars['Dist_pars']['Weighted_probs']]
Core_pars_vals                            = [pd.read_csv(x) for x in Core_pars['Dist_pars']['Weighted_probs']]
Core_pars['Dist_pars']['Weighted_probs']  = {}

for i in range(len(Core_pars_keys)):
    
    Core_pars['Dist_pars']['Weighted_probs'].setdefault(Core_pars_keys[i],[]).append(Core_pars_vals[i])


###########################################################################

### Get fire maps

###########################################################################

Core_pars['Fire_use']['bool'] = ''
Core_pars['Fire_use']['ba']   = ''

Fire_pars             = [s.replace('\\', '/') for s in file_list if "Fire use" in s]
bool_pars             = [s for s in Fire_pars if "bool.csv" in s]
ba_pars               = [s for s in Fire_pars if "ba.csv" in s]


bool_pars             = dict(zip([x[(Rlen+9):-9] for x in bool_pars], 
                                 [pd.read_csv(x) for x in bool_pars]))

ba_pars               = dict(zip([x[(Rlen+9):-7] for x in ba_pars], 
                                 [pd.read_csv(x) for x in ba_pars]))

Core_pars['Fire_use']['bool'] = bool_pars
Core_pars['Fire_use']['ba']   = ba_pars


### Fire escape
escape_pars                = [s.replace('\\', '/') for s in file_list if "Fire escape" in s]
escape_rate                = [pd.read_csv(s) for s in escape_pars if "pars" in s]
escape_pars                = [s for s in escape_pars if "tree" in s]


escape_dict                = {'Overall': escape_rate[0]}
escape_dict['fire_types']  = dict(zip([x[(Rlen+12):-16] for x in escape_pars], 
                                 [pd.read_csv(x) for x in escape_pars]))


Core_pars['Fire_escape'] = escape_dict

##############################

### Suppression

##############################

sup_pars              = [s.replace('\\', '/') for s in file_list if "Fire suppression" in s]
bool_pars             = [s for s in sup_pars if "bool.csv" in s]
ba_pars               = [s for s in sup_pars if "ba.csv" in s]


Core_pars['Fire_suppression'] = {'bool': pd.read_csv(bool_pars[0]), 
                                 'ba': pd.read_csv(ba_pars[0])}

###########################################################################

### Get maps

###########################################################################

Map_list = []

for path, subdirs, files in os.walk(Map_folder):
    for name in files:
        Map_list.append(os.path.join(path, name))

Maps       = [s.replace('\\', '/') for s in Map_list if ".nc" in s]
Mask       = [s.replace('\\', '/') for s in Map_list if "mask.csv" in s]
Area       = [s.replace('\\', '/') for s in Map_list if "Area.csv" in s]

Map_data = dict(zip([x[Mlen:-3] for x in Maps], 
            [nc.Dataset(Map_folder + x[Mlen:-3] + '.nc') for x in Maps]))

var_key  = zip([x for x in Map_data.values()], 
               [[x for x in y.variables.keys()][len(y.variables.keys()) -1 ] for y in Map_data.values()])

Map_data = dict(zip([x for x in Map_data.keys()], 
            [x[y][:] for x, y in var_key]))

Map_data['Mask'] = np.array(pd.read_csv(Mask[0])).reshape(27648)
Map_data['Area'] = np.array(pd.read_csv(Area[0])).reshape(27648)

###########################################################################

Map_data['Market_influence'] = Map_data['GDP'] * Map_data['Market_access'][0:26, :, :]
Map_data['Market.influence'] = Map_data['GDP'] * Map_data['Market_access'][0:26, :, :]
Map_data['HDI_GDP']          = np.log(Map_data['GDP'].data) * Map_data['HDI']
Map_data['WFI']              = (1/Map_data['TRI']) * Map_data['GDP']


### handle missing values in processed data
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
        
        Core_pars['Dist_pars']['Thresholds'][key][j] = pd.concat([Core_pars['Dist_pars']['Thresholds'][key][j][0:51] , 
                                                        Core_pars['Dist_pars']['Thresholds'][key][j][51:100]])
        
    for j in range(len(Core_pars['Dist_pars']['Probs'][key])):

        Core_pars['Dist_pars']['Probs'][key][j] = pd.concat([Core_pars['Dist_pars']['Probs'][key][j][0:51], 
                                                    Core_pars['Dist_pars']['Probs'][key][j][51:100]])


###########################################################################

### Seasonality of fire use

###########################################################################

Seasonality = [x for x in file_list if 'seasonality' in x]
Seasonality = [x.replace('\\', '/') for x in Seasonality if '.nc' in x]

Season_Map = dict(zip([x[(Rlen+17):-3] for x in Seasonality], 
            [nc.Dataset(x) for x in Seasonality]))

var_key  = zip([x for x in Season_Map.values()], 
               [[x for x in y.variables.keys()][len(y.variables.keys()) -1 ] for y in Season_Map.values()])

Seasonality = dict(zip([x for x in Season_Map.keys()], 
            [x[y][:] for x, y in var_key]))



import gc
gc.collect()
