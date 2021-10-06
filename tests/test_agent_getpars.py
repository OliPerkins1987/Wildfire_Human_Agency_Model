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
exec(open("test_setup.py").read())


from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree
from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.AFTs.afts import SOSH, Intense_arable
from model_interface.wham import WHAM


#########################################################################

### Load test data

#########################################################################

os.chdir(str(wd + '/test_data/AFTs').replace('\\', '/'))
Trans_frame   = pd.read_csv('Trans_pars.csv')
Intense_frame = pd.read_csv('Intense_arable_pars.csv')
SOSH_frame    = pd.read_csv('SOSH_pars.csv')

### Mock load up
Core_pars = {'AFT_dist': {}, 
             'Fire_use': {}} 

Core_pars['AFT_dist']['Cropland/Trans']   = Trans_frame
Core_pars['AFT_dist']['Cropland/Intense'] = Intense_frame
Core_pars['AFT_dist']['Sub_AFTs/Trans_Cropland'] = SOSH_frame

### Mock model
parameters = {
    
    'xlen': 144, 
    'ylen': 192,
    'AFTs': [SOSH, Intense_arable],
    'LS'  : [],
    'AFT_pars': Core_pars,
    'Maps'    : '',
    'timestep': 0,
    'theta'    : 0.1, 
    'bootstrap': False, 
    'Observers': {},
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
    


