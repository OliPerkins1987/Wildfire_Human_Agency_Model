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

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars
from Core_functionality.AFTs.agent_class import AFT, dummy_agent
from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Nonex

from model_interface.wham import WHAM


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
dummy_pars = {'AFT_dist': {}, 
             'Fire_use': {}, 
             'Dist_pars':{'Thresholds': {}, 
                          'Probs': {}}} 

dummy_pars['AFT_dist']['Xaxis/Forest']       = deepcopy(Dummy_frame)
dummy_pars['AFT_dist']['Xaxis/Other']        = deepcopy(Dummy_frame)

dummy_pars['Dist_pars']['Thresholds']['Xaxis/Forest']  = [pd.DataFrame(np.random.normal(8.5, 10, 10)), 
                                                              pd.DataFrame(np.random.normal(240, 10, 10))]

dummy_pars['Dist_pars']['Probs']['Xaxis/Forest']       = [pd.DataFrame(pd.Series([np.random.beta(1, 1) for x in range(10)]), 
                                                        columns = ['TRUE.']) for x in range(3)]

dummy_pars['Dist_pars']['Thresholds']['Xaxis/Other']  = dummy_pars['Dist_pars']['Thresholds']['Xaxis/Forest']

dummy_pars['Dist_pars']['Probs']['Xaxis/Other']       = dummy_pars['Dist_pars']['Probs']['Xaxis/Forest']




### Mock model
parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [],
    'LS'  : [Nonex],
    'AFT_pars': dummy_pars,
    'Maps'    : Map_data,
    'start_run': 0,
    'theta'    : 0.1, 
    'bootstrap': True, 
    'Observers': {},
    'reporters': [],
    'n_cores'  : 4
    
    }


##########################################################################

### tests

##########################################################################

def test_LS_boot():
    
    errors = []
    
    mod = WHAM(parameters)
    mod.setup()
    mod.ls.setup()
    mod.ls.get_pars(mod.p.AFT_pars)
    mod.ls.get_boot_vals(mod.p.AFT_pars)
    mod.ls.get_vals()
    
    if not np.array_equal(mod.ls[0].Dist_vals['Forest'], mod.ls[0].Dist_vals['Other']):
        
        errors.append("Bootstrapped X-axis predictions for Nonex shold be identical")
    
    probs = mod.ls[0].Dist_frame['Forest']['yprob.TRUE'][mod.ls[0].Dist_frame['Forest']['var'] == '<leaf>'].to_list()
    
    if not probs == [float(x.iloc[-1]) for x in mod.ls[0].boot_Dist_pars['Forest']['Probs']]:
        
        errors.append("Bootstrapped parameters not loaded properly")
    
    ### which values do not equal the mode?
    gt_thresh_1 = len(pd.concat([pd.Series(np.arange(0, x)) if x >= 1 else pd.Series(0) for x in mod.ls[0].boot_Dist_pars['Forest']['Thresholds'][0][0]]).unique())-1
    
    if not gt_thresh_1 == len(np.where(mod.ls[0].Dist_vals['Forest'] != stats.mode(np.array(mod.ls[0].Dist_vals['Forest']),
                                                                                           keepdims = False)[0])[0]):
    
        errors.append("Bootstrapped prediction error")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
