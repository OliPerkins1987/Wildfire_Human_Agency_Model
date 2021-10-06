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
from copy import deepcopy

random.seed(1987)

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

class multiple_agent(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Test'
        self.ls  = 'Test'
        
        self.sub_AFT = {'exists': True, 'kind': 'Multiple',  
                        'afr': ['Test', 'Test'], 'ls': ['Test', 'Test']}    
        



os.chdir(str(wd + '/test_data/AFTs').replace('\\', '/'))
Dummy_frame   = pd.read_csv('Dummy_pars.csv')
Dummy_dat     = nc.Dataset('Test.nc')
Dummy_dat     = Dummy_dat['Forest_frac'][:]
Dummy_dat2    = 27647 - np.arange(27648)
Map_data      = {'Test': Dummy_dat, 'Test2': Dummy_dat2}
Map_test      = np.array(pd.read_csv('Test_raster.csv'))

### Mock load up
dummy_pars = {'AFT_dist': {}, 
             'Fire_use': {}, 
             'Dist_pars':{'Thresholds': {}, 
                          'Probs': {}}} 

dummy_pars['AFT_dist']['Test/Test']          = Dummy_frame
dummy_pars['AFT_dist']['Sub_AFTs/Test_Test'] = deepcopy(Dummy_frame)
dummy_pars['AFT_dist']['Sub_AFTs/Test_Test'].columns = ['Unnamed: 0', 'var', 'n', 'dev', 'yval', 'splits.cutleft',
                                                        'splits.cutright', 'yprob.FALSE', 'multiple_agent']

dummy_pars['Dist_pars']['Thresholds']['Test/Test']  = [pd.DataFrame(np.random.normal(8.5, 10, 1)), 
                                                              pd.DataFrame(np.random.normal(240, 10, 1))]

dummy_pars['Dist_pars']['Probs']['Test/Test']       = [pd.DataFrame(pd.Series([np.random.beta(1, 1) for x in range(1)]), 
                                                        columns = ['TRUE.']) for x in range(3)]

dummy_pars['Dist_pars']['Thresholds']['Sub_AFTs/Test_Test']  = [pd.DataFrame(np.random.normal(8.5, 10, 10)), 
                                                              pd.DataFrame(np.random.normal(240, 10, 10))]

dummy_pars['Dist_pars']['Probs']['Sub_AFTs/Test_Test']       = [pd.DataFrame(pd.Series([np.random.beta(1, 1) for x in range(10)]), 
                                                        columns = ['multiple_agent']) for x in range(3)]



### Mock model
parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [multiple_agent],
    'LS'  : [],
    'AFT_pars': dummy_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1, 
    'bootstrap': True, 
    'Observers': {},
    'reporters': []
    
    }


##########################################################################

### tests

##########################################################################

def test_AFT_boot():
    
    errors = []
    
    mod = WHAM(parameters)
    mod.setup()
    mod.agents.sub_compete()
    
    probs = mod.agents[0].AFT_frame[1]['multiple_agent'][mod.agents[0].AFT_frame[1]['var'] == '<leaf>'].to_list()
    
    if not probs == [float(x.iloc[-1]) for x in mod.agents[0].boot_AFT_pars[1]['Probs']]:
        errors.append("Bootstrapped parameters not loaded properly")
    
    ### which values do not equal the mode?
    gt_thresh_1 = len(pd.concat([pd.Series(np.arange(0, x)) if x >= 1 else pd.Series(0) for x in mod.agents[0].boot_AFT_pars[0]['Thresholds'][0][0]]).unique())-1
    
    if not gt_thresh_1 == len(np.where(mod.agents[0].AFT_vals[0] != stats.mode(np.array(mod.agents[0].AFT_vals[0]))[0][0])[0]):
    
        errors.append("Bootstrapped prediction error")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
