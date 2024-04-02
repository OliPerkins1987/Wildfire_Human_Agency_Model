# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:25:15 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
from scipy import stats
import netCDF4 as nc
import os
import random
from dask.distributed import Client
import agentpy as ap


from model_interface.wham import WHAM
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap


#########################################################################

### Load test data

#########################################################################

os.chdir((str(os.getcwd()) + '/test_data/AFTs').replace('\\', '/'))
Dummy_frame   = pd.read_csv('Dummy_pars.csv')
Dummy_dat     = nc.Dataset('Test.nc')
Dummy_dat     = Dummy_dat['Forest_frac'][:]
Dummy_dat2    = 27647 - np.arange(27648)
Map_data      = {'Test': Dummy_dat, 'Test2': Dummy_dat2}
Map_test      = np.array(pd.read_csv('Test_raster.csv'))
Map_data['Area'] = np.array([1]*27648)

### Mock load up
Core_pars = {'AFT_dist': {}, 
             'Fire_use': {}} 

Core_pars['AFT_dist']['Test/Test']          = Dummy_frame
Core_pars['AFT_dist']['Sub_AFTs/Test_Test'] = Dummy_frame

### Mock model
parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [dummy_agent, dummy_agent],
    'LS'  : [],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'start_run': 0,
    'theta'    : 0.1, 
    'bootstrap': False, 
    'Observers': {}, 
    'reporters': []
    
    }


##########################################################################

### tests

##########################################################################

@pytest.mark.usefixtures("mod_pars")
def test_prediction_frame(mod_pars):  
    
    errors = []
    
    ### setup model
    mod                  = WHAM(mod_pars)
    mod.p.bootstrap      = True
    mod.p.numb_bootstrap = 2
    mod.timestep         = 0
    mod.xlen             = mod_pars['xlen']
    mod.ylen             = mod_pars['ylen']
    
    
