# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:17:04 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os
from copy import deepcopy

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

os.chdir((wd[0:-6] + '/src/data_import'))
exec(open("local_load_up.py").read())

os.chdir(wd)
exec(open("setup_full.py").read())

######################
### Run annual model
######################

parameters['Seasonality']  = False
parameters['escaped_fire'] = False
parameters['reporters']    = ['Managed_fire']
parameters['write_annual'] = False

mod = WHAM(parameters)

### setup
mod.setup()

### go
mod.go()

mod_annual = deepcopy(mod.results['Managed_fire'][0]['Total'])

#######################
### Run model monthly
#######################

parameters['Seasonality'] = True
    
mod = WHAM(parameters)

### setup
mod.setup()

### go
mod.go()

##################################

### tests

##################################

def test_seasonality_mean():
    
    seasonal = np.nansum(mod.results['Managed_fire'][0]['Total'], axis = 0)
    
    assert pytest.approx(np.nanmean(mod_annual)) == np.nanmean(seasonal)


def test_seasonality_quantiles():
    
    seasonal = np.nansum(mod.results['Managed_fire'][0]['Total'], axis = 0)
    quants   = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    
    
    assert pytest.approx(np.nanquantile(mod_annual, quants)) == np.nanquantile(seasonal, quants)
