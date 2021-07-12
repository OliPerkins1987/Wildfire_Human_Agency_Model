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
             'Fire_use': ''} #empty dict to house files

##########################################################################

### Get AFT distribution parameters

##########################################################################

AFT_dist              = [s for s in matching if "AFT Distribution/Trees" in s]
Core_pars['AFT_dist'] = [s for s in AFT_dist if "Tree_frame.csv" in s]

Core_pars_keys        = [x[51:-15] for x in Core_pars['AFT_dist']]
Core_pars_vals        = [read_shpt_data(x) for x in Core_pars['AFT_dist']]
Core_pars['AFT_dist'] = dict(zip(Core_pars_keys, Core_pars_vals))

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

Map_data['HDI_GDP']          = np.log(Map_data['GDP']) * Map_data['HDI']
Map_data['Market_influence'] = Map_data['GDP'] * Map_data['Market_access'][0:26, :, :]
Map_data['Market.influence'] = Map_data['GDP'] * Map_data['Market_access'][0:26, :, :]
Map_data['WFI']              = (1/Map_data['TRI']) * Map_data['GDP']

