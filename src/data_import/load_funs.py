# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:35:00 2024

@author: Oli
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
import os
import re
from copy import deepcopy

def mk_par_dict(dat, filt, name_key, kind = 'single', filetype = 'csv'):
    
    '''
    utility to load in files
    dat: a list of file paths
    filt: a character string on which to filter dat
    name_key: a set of integers on which to extract keys for the resulting dictionary
    kind: one of 'single' - where the values are a single object (e.g. a pandas) and
                 'multiple' - where values are a list of objects
    filetype: one of 'csv' (array-like) or 'nc' (spatial)
    
    '''
    
    pars_dict = [s for s in dat if filt in s]
    
    if filetype == 'csv':
        
        pars_keys        = [x[(name_key[0]+name_key[1]):-name_key[2]] for x in pars_dict]
        pars_vals        = [pd.read_csv(x) for x in pars_dict]
            
        if kind == 'single':
            
            pars_dict = dict(zip(pars_keys, pars_vals))
            
        elif kind == 'multiple':
            
            pars_dict = {}

            for i in range(len(pars_keys)):
    
                pars_dict.setdefault(pars_keys[i],[]).append(pars_vals[i])
    
    return(pars_dict)
    
