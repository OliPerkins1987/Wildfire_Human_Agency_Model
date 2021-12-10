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
wd = os.getcwd().replace('\\', '/')
os.chdir((wd[0:-6] + '/src/data_import'))
exec(open("local_load_up.py").read())

os.chdir(wd)
exec(open("setup_full.py").read())

### RData
os.chdir(str(wd + '/test_data/R_outputs').replace('\\', '/'))
Defor_R = pd.read_csv('Deforestation_2014.csv')/100

### set to end of run
parameters['start_run'] = 24
parameters['end_run']   = 24   

### model set up
mod = WHAM(parameters)

### setup
mod.setup()

### go
mod.go()


def test_defor():
    
    errors = []
    
    Defor_WHAM = pd.Series(mod.Observers['deforestation'][0].VC_vals.reshape(144*192))
    Defor_WHAM[np.isnan(Defor_R).iloc[:, 0]] = np.nan 

    Defor_frame = pd.concat([Defor_WHAM, pd.Series(Defor_R.iloc[:, 0])], axis = 1)
    Defor_frame.columns = ['Wham', 'R']
    Defor_frame['Diff'] = Defor_frame['Wham'] - Defor_frame['R']
    
    if not (np.nanmean(Defor_frame['Wham']) == pytest.approx(np.nanmean(Defor_frame['R']), 0.1)):
        
        errors.append("Deforestation does not match baseline calculation")
        
    if not (np.nanmax(Defor_frame['Wham']) == pytest.approx(np.nanmax(Defor_frame['R']), 0.33)):
        
        errors.append("Deforestation does not match baseline calculation")
    
    
    
    