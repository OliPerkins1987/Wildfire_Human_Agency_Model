# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:25:36 2021

@author: Oli
"""


import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
exec(open("test_setup.py").read())

os.chdir(real_dat_path)
exec(open("local_load_up.py").read())

os.chdir(str(test_dat_path) + '\R_outputs')
R_results = pd.read_csv('Background_rate.csv')

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation

from model_interface.wham import WHAM
from Core_functionality.top_down_processes.background_ignitions import background_rate

parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [],
    'LS'  : [],
    'Fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation'}, 
    'Observers': {'background_rate': background_rate},
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'end_run' : 0,
    'reporters': [],
    'theta'    : 0.1,
    'bootstrap': False
    
    }


mod = WHAM(parameters)

### setup
mod.setup()

### ignite
mod.Observers['background_rate'].ignite()


def test_backgroundrate_fundamentals():
    
    errors = []
    summary= mod.Observers['background_rate'][0].Fire_vals.describe()
    
    
    if not mod.Observers['background_rate'].Fire_vals[0].iloc[0] == 0.0:
        
        errors.append("Incorrect ordering of background rate predictions")
    
    if not np.nanmedian(mod.Observers['background_rate'].Fire_vals[0]) == 0.0:
        
        errors.append("Incorrect background_rate predictions")
    
    if not np.nanmax(mod.Observers['background_rate'].Fire_vals[0]) <= 1.0:
        
        errors.append("Incorrect background_rate predictions")
       
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    

def test_backgroundrate_fundamentals():

    errors = []
    R_results['Test_1990']     = mod.Observers['background_rate'].Fire_vals[0]
    R_results['Test_1990']     = [x if y >= 0 else np.nan for x, y in zip(R_results['Test_1990'], R_results['Year_1990'])]
    R_results['Delta']         = R_results.Test_1990 - R_results.Year_1990
    
    ### add anciliary variables
    R_results['NPP']           = Map_data['NPP'][0, :, :].data.reshape(27648)
    R_results['Market_access'] = Map_data['Market_access'][0, :, :].data.reshape(27648)

    Wrong = R_results[np.abs(R_results.Delta) > 0.01]    
    
    if not Wrong.shape[0] < 400:
        
        errors.append("Python predictions do not match R calculations")
    
    if not np.abs(np.nanmax(R_results.Test_1990) - np.nanmax(R_results.Year_1990)) < 0.001:
        
        errors.append("Python predictions do not match R calculations")

    if not np.abs(np.nanmean(R_results.Test_1990) - np.nanmean(R_results.Year_1990)) < 0.002:
        
        errors.append("Python predictions do not match R calculations")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))

