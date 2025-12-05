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
from dask.distributed import Client
import agentpy as ap


from model_interface.wham import WHAM
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_numpy
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap

#########################################################

### load data

#########################################################

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')
os.chdir(str(wd + '/test_data/AFTs').replace('\\', '/'))
Cons_frame    = pd.read_csv('Conservationist_pars.csv')



##########################################################################

### tests

##########################################################################

@pytest.mark.usefixtures("mod_pars")
def test_tree_predict(mod_pars):  
    
    errors = []
    
    mod                  = WHAM(mod_pars)
    mod.setup()
    mod.p.bootstrap      = False
    mod.p.numb_bootstrap = 2
    mod.timestep         = 0
    
    ###################################################################################
    
    ### Conservationist AFT
    
    ###################################################################################
    
    Cons_pred = np.select([(mod_pars['Maps']['Pop'][0, :, ].mask) == True, 
                           mod_pars['Maps']['NPP_mountain'][0, :, ].data < float(Cons_frame['splits.cutleft'][0][1:]), 
                       mod_pars['Maps']['Pop'][0, :, :].data < float(Cons_frame['splits.cutleft'][2][1:]), 
                       mod_pars['Maps']['Pop'][0, :, :].data >= float(Cons_frame['splits.cutleft'][2][1:])], 
                    [np.array(0), np.array(Cons_frame['yprob.TRUE'][1]), np.array(Cons_frame['yprob.TRUE'][3]), np.array(Cons_frame['yprob.TRUE'][4])], 
                    default = 0)
    
    aft = mod.agents[19]
    
    ##### test prediction #####
        
    ### gather numpy arrays of predictor variables
    aft.Dist_dat  = [aft.model.p.Maps[x][aft.model.timestep, :, :] if len(aft.model.p.Maps[x].shape) == 3 else aft.model.p.Maps[x] for x in aft.Dist_vars]

    ### combine numpy arrays to single pandas       
    aft.Dist_dat  = np.array([x.reshape(aft.model.p.xlen*aft.model.p.ylen).data for x in aft.Dist_dat]).transpose()
        
    ### do prediction
    aft.Dist_vals = predict_from_tree_numpy(dat = aft.Dist_dat, 
                              tree = Cons_frame, split_vars = aft.Dist_vars, struct = aft.Dist_struct,
                               prob = 'yprob.TRUE', skip_val = -1e+10, na_return = 0)   
    
    #############################
    
    ### test allocation correct
    
    #############################
    
    if not(np.nanmax(aft.Dist_vals - Cons_pred.reshape(mod.xlen*mod.ylen)) == pytest.approx(0, abs = 0.0001)):
            
            errors.append("Tree prediction error: conservationist")
            
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
    
    ###################################################################################
    
    ### HG AFT
    
    ###################################################################################
    
    aft = mod.agents[16]
    HG_frame = aft.Dist_frame
    
    ### Read this in as a .csv
    
    HG_pred = np.select([(mod_pars['Maps']['Pop'][0, :, ].mask) == True, 
                           mod_pars['Maps']['Market.Inf'][0, :, ].data >= float(HG_frame['splits.cutleft'][0][1:]), 
                       mod_pars['Maps']['ET_pop'][0, :, :].data >= float(HG_frame['splits.cutleft'][1][1:]), 
                       mod_pars['Maps']['HDI'][0, :, :].data < float(HG_frame['splits.cutleft'][2][1:]), 
                       mod_pars['Maps']['HDI'][0, :, :].data >= float(HG_frame['splits.cutleft'][2][1:])], 
                    [np.array(0), np.array(HG_frame['yprob.TRUE'][6]), np.array(HG_frame['yprob.TRUE'][5]), 
                     np.array(HG_frame['yprob.TRUE'][3]), np.array(HG_frame['yprob.TRUE'][4])], 
                    default = 0)
    
    aft = mod.agents[16]
    
    ##### test prediction #####
        
    ### gather numpy arrays of predictor variables
    aft.Dist_dat  = [aft.model.p.Maps[x][aft.model.timestep, :, :] if len(aft.model.p.Maps[x].shape) == 3 else aft.model.p.Maps[x] for x in aft.Dist_vars]

    ### combine numpy arrays to single pandas       
    aft.Dist_dat  = np.array([x.reshape(aft.model.p.xlen*aft.model.p.ylen).data for x in aft.Dist_dat]).transpose()
        
    ### do prediction
    aft.Dist_vals = predict_from_tree_numpy(dat = aft.Dist_dat, 
                              tree = aft.Dist_frame, split_vars = aft.Dist_vars, struct = aft.Dist_struct,
                               prob = 'yprob.TRUE', skip_val = -1e+10, na_return = 0)   
    
    #############################
    
    ### test allocation correct
    
    #############################
    
    if not(np.nanmean(aft.Dist_vals) - np.nanmean(HG_pred) == pytest.approx(0, abs = 0.01)):
            
            errors.append("Tree prediction error: HG")
            
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    