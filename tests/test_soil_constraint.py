# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:05:09 2021

@author: Oli
"""


import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

dat = np.arange(27648).reshape(144, 192)

os.chdir(str(wd + '/test_data/Fire').replace('\\', '/'))
Soil= nc.Dataset('Baresoil.nc')
Soil= Soil['Soil'][:]


### Calculate soil constraint
Soil = 1 - (Soil * (Soil>= 0.1325)) 
dat_r  = dat * Soil
        

def test_soil_calc():
    
    assert len(np.where(np.logical_and(Soil > (1-0.1325), Soil <1))[0]) == 0

def test_soil_restrict():
    
    assert len(np.where(dat < dat_r)[0]) == 0



