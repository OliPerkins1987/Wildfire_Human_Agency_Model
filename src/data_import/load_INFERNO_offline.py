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


##########################################################################

### import data

##########################################################################

### Set these to your local directories!

root       = r'C:\Users\Oli\Documents\PhD\wham_coupled\JULES_inputs\Annual'
os.chdir(root)

Lightning= nc.Dataset('Lightning_fires_annual.nc')
Lightning= Lightning['variable'][:]

PFT      = nc.Dataset('PFTsJULES_output_2.nc')['variable'][:]
PFT_ba   = nc.Dataset('PFT_ba.nc')['variable'][:]

Flammability = nc.Dataset('FlammabilityJULES_output_2.nc')['variable'][:]

INFERNO    = {'Lightning_fires' : Lightning[1, :, :], 
              'PFT_ba'          : PFT_ba, 
              'PFT'             : PFT[0:13, :, :], 
              'Bare_soil'       : PFT[15, :, :],
              'Flammability'    : Flammability}

del([root, Lightning, PFT, PFT_ba, Flammability])
