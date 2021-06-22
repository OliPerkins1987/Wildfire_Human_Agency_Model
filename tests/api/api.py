# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:16:40 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import sharepy
import io
import pandas as pd
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime


env_path = Path("..") / ".env"  # move up one directory
load_dotenv(dotenv_path=env_path)
print(os.getenv("SERVER"))


from data_import.api.Access_sharepoint import read_shpt_data, shpt_file_list, write_shpt_data

###########################################################################
### import data
###########################################################################

### local
test_file = pd.read_csv(r'C:\Users\Oli\Documents\PhD\wham\tests\Test_data\Sharepoint\test.csv')
    
### remote

print(os.getenv("SERVER"))

file_list = shpt_file_list()
matching = [s for s in file_list if "Documents/wham_files/test.csv" in s]

current_files = {'dat_test': matching[0]}

dat = dict(zip([x for x in current_files.keys()], 
               [read_shpt_data(x) for x in current_files.values()]))


def not_null_read_test():
    
    assert(len(matching) > 0)

def csv_load_test():
   
    assert(np.array_equal(dat['dat_test']['yprob.FALSE'], test_file['yprob.FALSE']))

### write data

