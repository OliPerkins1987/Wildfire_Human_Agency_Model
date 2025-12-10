# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:25:15 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree_numpy
from Core_functionality.AFTs.agent_class import AFT, dummy_agent, multiple_agent
from model_interface.wham import WHAM


#########################################################################

### Load test data

#########################################################################

os.chdir(str(wd + '/test_data/AFTs').replace('\\', '/'))
Dummy_frame   = pd.read_csv('Dummy_pars.csv')
Dummy_frame2  = pd.read_csv('Dummy_AFT_pars.csv')
Dummy_dat     = nc.Dataset('Test.nc')
Dummy_dat     = Dummy_dat['Forest_frac'][:]
Map_data      = {'Test': Dummy_dat}
Map_test      = np.array(pd.read_csv('Test_raster.csv'))

LFS_test      = [np.arange(0, 1, step = 1/27648).reshape(144, 192)] * 4


### Mock load up
Core_pars = {'AFT_dist': {}, 
             'Fire_use': {}} 

Core_pars['AFT_dist']['Test/Test']          = Dummy_frame
Core_pars['AFT_dist']['Sub_AFTs/Test_Test'] = Dummy_frame2

### Mock model
parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [dummy_agent],
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

def test_AFT_addition():
    
    a = dummy_agent 
    
    ### setup 1st agent for test
     
    a.afr = 'Test'
    a.ls  = 'Test'
        
    a.sub_AFT = {'exists': True, 'kind': 'Addition',  
                        'afr': 'Test', 'ls': 'Test'}    
    a.AFT_frame  = Dummy_frame2
    a.AFT_struct = define_tree_links(a.AFT_frame)
    a.AFT_vars   = [x for x in a.AFT_frame.iloc[:,1].tolist() if x != '<leaf>']
    
    parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [a],
    'LS'  : [],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1
    
    }
    
    if a.sub_AFT['exists'] == True:
        
        if a.sub_AFT['kind'] != 'Multiple':    
        
            ### gather correct numpy arrays 4 predictor variables
            a.AFT_dat   = [Map_data[x][0, :, :] if len(Map_data[x].shape) == 3 else Map_data[x] for x in a.AFT_vars]
        
            ### combine numpy arrays to single pandas       
            a.AFT_dat   = np.array([x.reshape(parameters['xlen'] * parameters['ylen']).data for x in a.AFT_dat]).transpose()
            
            ### do prediction
            a.AFT_vals  = predict_from_tree_numpy(dat = a.AFT_dat, 
                              tree = a.AFT_frame, split_vars = a.AFT_vars, struct = a.AFT_struct,
                               prob = "dummy_agent", skip_val = -1e+10, na_return = 0)
        

    mod = WHAM(parameters)
    mod.agents = [a]
    mod.ylen = mod.p.ylen
    mod.xlen = mod.p.xlen
    mod.LFS = {'Test':{'Test':np.array([0.5]*27648).reshape(mod.ylen, mod.xlen)}}
    mod.allocate_AFT()

    vals = [pd.Series(x.reshape(27648)).value_counts() for x in mod.AFT_scores.values()]

    errors = []
    
    if not np.array_equal(vals[0].values, np.array([27233,   415])):
        errors.append("Addition AFT distribution failed")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))



def test_AFT_multiple():

    ### setup 2nd agent for test
    b     = multiple_agent    
    b.afr = 'Test'
    b.ls  = 'Test'
    
    b.sub_AFT = {'exists': True, 'kind': 'Multiple',  
                        'afr': ['Test', 'Test'], 'ls': ['Test', 'Test']}      
    
    
    b.AFT_frame  = [Dummy_frame2, Dummy_frame2]
    b.AFT_struct = [define_tree_links(b.AFT_frame[0]), define_tree_links(b.AFT_frame[1])]
    b.AFT_vars   = [[x for x in b.AFT_frame[0].iloc[:,1].tolist() if x != '<leaf>'], 
                   [x for x in b.AFT_frame[1].iloc[:,1].tolist() if x != '<leaf>']]
    
    b.AFT_dat  = []
    b.AFT_vals = []
    
    parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [b],
    'LS'  : [],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1
    
    }
    
    for i in range(len(b.sub_AFT['afr'])): 
            
        ### gather correct numpy arrays 4 predictor variables
        b.AFT_dat.append([Map_data[x][i, :, :] if len(Map_data[x].shape) == 3 else (i*1000) - Map_data[x] for x in b.AFT_vars[i]])
        
        ### combine numpy arrays to single pandas
        b.AFT_dat[i]   = np.array([x.reshape(parameters['xlen'] * parameters['ylen']).data for x in b.AFT_dat[i]]).transpose()
        
        ### do prediction
        b.AFT_vals.append(predict_from_tree_numpy(dat = b.AFT_dat[i], 
                          tree = b.AFT_frame[i], split_vars = b.AFT_vars[i], struct = b.AFT_struct[i],
                           prob = "dummy_agent", skip_val = -1e+10, na_return = 0))


    
    
    mod = WHAM(parameters)
    mod.agents = [b]
    mod.ylen = mod.p.ylen
    mod.xlen = mod.p.xlen
    mod.LFS = {'Test':{'Test':np.array([0.5]*27648).reshape(mod.ylen, mod.xlen)}}
    mod.allocate_AFT()

    vals = [pd.Series(x.reshape(27648)).value_counts() for x in mod.AFT_scores.values()]

    errors = []
    
    if not np.array_equal(vals[0].values, np.array([27064,   584])):
        errors.append("Multiple fractions AFT distribution failed")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))

