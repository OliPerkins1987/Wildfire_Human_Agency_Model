# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:16:40 2021

@author: Oli
"""

import pandas as pd
import os

from data_import.api.Access_sharepoint import read_shpt_data, shpt_file_list, write_shpt_data

### import data

file_list = shpt_file_list()
matching = [s for s in file_list if "wham_files" in s]





current_files = {'Attend': matching[1]}

dat = dict(zip([x for x in current_files.keys()], 
               [read_shpt_data(x) for x in current_files.values()]))

