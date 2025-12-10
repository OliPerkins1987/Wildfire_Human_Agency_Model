# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:25:15 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree_numpy
from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.AFTs.arable_afts import SOSH, Intense_arable
from model_interface.wham import WHAM


#########################################################################

### Load test data

#########################################################################

from data_import.local_load_up_func import load_local_data
Core_pars, Map_data, Seasonality = load_local_data()

os.chdir(str(wd + '/test_data/AFTs').replace('\\', '/'))
Trans_frame   = pd.read_csv('Trans_pars.csv')
Intense_frame = pd.read_csv('Intense_arable_pars.csv')
SOSH_frame    = pd.read_csv('SOSH_pars.csv')

### Mock model
parameters = {
    
    'xlen': 144, 
    'ylen': 192,
    'start_run': 0,
    'end_run' : 0,
    
    'AFTs': [SOSH, Intense_arable],
    'LS'  : [],
    'Observers': {},
    
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    
    ### Fire parameters
    'Fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation', 'defor': 'Vegetation'},

    'Seasonality'  : False, 
    'escaped_fire' : False,

    'theta'    : 0.1, 
    'bootstrap': False, 
    
    'reporters': []
    
    
    
    }


##########################################################################

### tests

##########################################################################

def test_LFS_pars_load():
    
    mod = WHAM(parameters)
    mod.setup()
    mod.agents.setup()
    mod.agents.get_pars(mod.p.AFT_pars)
    
    assert(np.array_equal(mod.agents.Dist_frame[0]['yprob.TRUE'], 
            Trans_frame['yprob.TRUE']))

def test_AFT_pars_load():
    
    errors = []
    
    mod = WHAM(parameters)
    mod.setup()
    mod.agents.setup()
    mod.agents.get_pars(mod.p.AFT_pars)
    
    if not (np.array_equal(mod.agents.AFT_frame[0]['SOSH'], SOSH_frame['SOSH'])):
        errors.append("AFT parameters not loaded correctly")
    
    if not (mod.agents.AFT_frame[1] == 'None'):
        errors.append("AFT parameters found where none expected")
        
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    


