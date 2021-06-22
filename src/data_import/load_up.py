# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:16:40 2021

@author: Oli
"""

import sharepy
import io
import pandas as pd
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

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

Core_pars_keys  = [x[51:-15] for x in Core_pars['AFT_dist']]
Core_pars_vals  = [read_shpt_data(x)  for x in Core_pars['AFT_dist']]
Core_pars['AFT_dist']    = dict(zip(Core_pars_keys, Core_pars_vals))


