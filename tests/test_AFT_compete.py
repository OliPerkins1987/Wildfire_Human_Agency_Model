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
exec(open("test_setup.py").read())

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree
from Core_functionality.AFTs.agent_class import AFT, dummy_agent, multiple_agent
from model_interface.wham import WHAM


#########################################################################

### Load test data

#########################################################################

os.chdir(str(test_dat_path) + '\AFTs')
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
    'timestep': 0,
    'theta'    : 0.1, 
    'bootstrap': False, 
    'Observers': {},
    'reporters': []
    
    }


##########################################################################

### tests

##########################################################################

def test_AFT_sub_compete():
    
    ### setup agent for test
    a = dummy_agent  
    a.afr = 'Test'
    a.ls  = 'Test'
        
    a.sub_AFT = {'exists': True, 'kind': 'Addition',  
                        'afr': 'Test', 'ls': 'Test'}    
    a.AFT_frame  = Dummy_frame2
    a.AFT_struct = define_tree_links(a.AFT_frame)
    a.AFT_vars   = [x for x in a.AFT_frame.iloc[:,1].tolist() if x != '<leaf>']
    
    if a.sub_AFT['exists'] == True:
        
        if a.sub_AFT['kind'] != 'Multiple':    
        
            ### gather correct numpy arrays 4 predictor variables
            a.AFT_dat   = [Map_data[x][0, :, :] if len(Map_data[x].shape) == 3 else Map_data[x] for x in a.AFT_vars]
        
            ### combine numpy arrays to single pandas       
            a.AFT_dat   = pd.DataFrame.from_dict(dict(zip(a.AFT_vars, 
                                  [x.reshape(27648).data for x in a.AFT_dat])))
            
            ### do prediction
            a.AFT_vals  = a.AFT_dat.apply(predict_from_tree, 
                                 axis = 1, tree = a.AFT_frame, struct = a.AFT_struct, 
                                  prob = 'dummy_agent', skip_val = -3.3999999521443642e+38, na_return = 0)
            

    assert(np.array_equal(a.AFT_vals.value_counts().values, np.array([27233,   415])))


def test_AFT_sub_compete_multiple():
    
    ### setup agent for test
    a     = multiple_agent    
    a.afr = 'Test'
    a.ls  = 'Test'
        
    a.sub_AFT = {'exists': True, 'kind': 'Multiple',  
                        'afr': ['Test', 'Test'], 'ls': ['Test', 'Test']}      
    
    
    a.AFT_frame  = [Dummy_frame2, Dummy_frame2]
    a.AFT_struct = [define_tree_links(a.AFT_frame[0]), define_tree_links(a.AFT_frame[1])]
    a.AFT_vars   = [[x for x in a.AFT_frame[0].iloc[:,1].tolist() if x != '<leaf>'], 
                   [x for x in a.AFT_frame[1].iloc[:,1].tolist() if x != '<leaf>']]
    
    a.AFT_dat  = []
    a.AFT_vals = []
                
    for i in range(len(a.sub_AFT['afr'])): 
            
        ### gather correct numpy arrays 4 predictor variables
        a.AFT_dat.append([Map_data[x][i, :, :] if len(Map_data[x].shape) == 3 else (i*1000) - Map_data[x] for x in a.AFT_vars[i]])
        
        ### combine numpy arrays to single pandas
        a.AFT_dat[i]   = pd.DataFrame.from_dict(dict(zip(a.AFT_vars[i], 
                                         [x.reshape(27648).data for x in a.AFT_dat[i]])))
            
        ### do prediction
        a.AFT_vals.append(a.AFT_dat[i].apply(predict_from_tree, 
                                         axis = 1, tree = a.AFT_frame[i], struct = a.AFT_struct[i], 
                                         prob = 'dummy_agent', skip_val = -3.3999999521443642e+38, na_return = 0))
            

    errors = []

    assert(np.array_equal(a.AFT_vals[i].value_counts().values, np.array([27064, 584])))











