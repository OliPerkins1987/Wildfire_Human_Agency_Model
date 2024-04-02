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

from data_import.load_funs import mk_par_dict

##########################################################################

### import data

##########################################################################

### set switches for data to import
### Addd to config file?
Dist = True
Maps = True
Fire = True
Nfer = True

### Set these to your local directories!

root       = r'C:/Users/Oli/Documents/PIES/WHAMv2/mod_data/'
Map_folder = root + r'Forcing/'

Rlen       = len(root)
Mlen       = len(Map_folder)

file_list = []

for path, subdirs, files in os.walk(root):
    for name in files:
        file_list.append(os.path.join(path, name))

file_list = [s.replace('\\', '/') for s in file_list]

#empty dict to house files
Core_pars = {'AFT_dist': '',
             'Dist_pars': {'Thresholds': '', 
             'Probabilities': ''},
             'Fire_use': {},
             'Fire_suppression':'',
             'Fire_escape': {},
             'Nfer_use': {}} 

##########################################################################

### Get AFT distribution parameters

##########################################################################

if Dist == True:

    ### Distribution parameters
    AFT_dist              = [s.replace('\\', '/') for s in file_list if "/Distribution" in s]

    ### Tree structures
    Core_pars['AFT_dist'] = mk_par_dict(dat = AFT_dist, filt = 'Tree_frame', name_key = [Rlen, 24, 15])

    ### Thresholds
    Core_pars['Dist_pars']['Thresholds'] = mk_par_dict(dat = AFT_dist, filt = 'thresholds', 
                                                    kind = 'multiple', name_key = [Rlen, 24, 17])

    ### Probs
    Core_pars['Dist_pars']['Probs']  = mk_par_dict(dat = AFT_dist, filt = 'probs', 
                                               kind = 'multiple', name_key = [Rlen, 24, 12])

###########################################################################

### Get AFT fire use pars

###########################################################################

if Fire == True:

    Core_pars['Fire_use']['bool'] = ''
    Core_pars['Fire_use']['ba']   = ''

'''

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


'''

###########################################################################

### Get AFT Nfer use parameters

###########################################################################

if Nfer == True:

    Core_pars['Nfer_use']['tree'] = ''
    Core_pars['Nfer_use']['lm']   = ''


    Nfer_pars             = [s.replace('\\', '/') for s in file_list if "Nfer_use" in s]
    tree_pars             = [s for s in Nfer_pars if "tree.csv" in s]
    lm_pars               = [s for s in Nfer_pars if "lm.csv" in s]

    tree_pars             = dict(zip([x[(Rlen+25):-9] for x in tree_pars], 
                                 [pd.read_csv(x) for x in tree_pars]))

    lm_pars               = dict(zip([x[(Rlen+23):-7] for x in lm_pars], 
                                 [pd.read_csv(x) for x in lm_pars]))

    Core_pars['Nfer_use']['tree'] = tree_pars
    Core_pars['Nfer_use']['lm']   = lm_pars

###########################################################################

### Get forcing maps

###########################################################################

if Maps == True:

    Map_list = []

    for path, subdirs, files in os.walk(Map_folder):
        for name in files:
            Map_list.append(os.path.join(path, name))

    Maps       = [s.replace('\\', '/') for s in Map_list if ".nc" in s]
    Mask       = [s.replace('\\', '/') for s in Map_list if "Mask.csv" in s]
    Area       = [s.replace('\\', '/') for s in Map_list if "Area.csv" in s]

    Map_data = dict(zip([x[Mlen:-3] for x in Maps], 
            [nc.Dataset(Map_folder + x[Mlen:-3] + '.nc') for x in Maps]))

    var_key  = zip([x for x in Map_data.values()], 
               [[x for x in y.variables.keys()][len(y.variables.keys()) -2 ] for y in Map_data.values()])

    Map_data = dict(zip([x for x in Map_data.keys()], 
            [x[y][:] for x, y in var_key]))

    shp      = Map_data[[x for x in Map_data.keys()][0]].shape[1] * Map_data[[x for x in Map_data.keys()][0]].shape[2]
    Map_data['Mask'] = np.array(pd.read_csv(Mask[0])).reshape(shp)
    Map_data['Area'] = np.array(pd.read_csv(Area[0])).reshape(shp)

###########################################################################

### make auxiliaries / convolutions

###########################################################################

    Map_data['Market.Inf']       = Map_data['GDP'] * Map_data['MA']
    Map_data['HDI_GDP']          = np.log(Map_data['GDP']) * Map_data['HDI']
    Map_data['W_flat']           = (1/Map_data['TRI']) * Map_data['GDP']


    ### handle missing values in processed data
    for i in range(Map_data['HDI_GDP'].shape[0]):
    
        Map_data['HDI_GDP'].data[i,:, :] = np.select([np.isnan(Map_data['HDI_GDP'].data[i,:, :]) == False], 
                                         [Map_data['HDI_GDP'].data[i, :, :]], default = -3.3999999521443642e+38)
    
        Map_data['W_flat'].data[i,:, :]  = np.select([Map_data['W_flat'].data[i,:, :] != 0], 
                                         [Map_data['W_flat'].data[i, :, :]], default = -3.3999999521443642e+38)



import gc
gc.collect()
