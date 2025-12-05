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

#########################################################################

### Load test data

#########################################################################


##########################################################################

### tests

##########################################################################

@pytest.mark.usefixtures("mod_pars")
def test_ls_prediction(mod_pars):  
    
    errors = []
    
    ### setup model
    mod                  = WHAM(mod_pars)
    mod.p.bootstrap      = True
    mod.p.numb_bootstrap = 2
    mod.timestep         = 0
    mod.xlen             = mod_pars['xlen']
    mod.ylen             = mod_pars['ylen']
    
    mod.ls     = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.LS]])
    
    ### setup agents
    mod.ls.setup()
    mod.ls.get_pars(mod.p.AFT_pars)
    mod.ls.get_boot_vals(mod.p.AFT_pars)
        
    boot_frame = {}
    
    ### run agent actions
    for ls in mod.ls:
        
        if ls.dist_method == 'Competition':
        
            ls.Dist_vals = []
            
            ### gather correct numpy arrays 4 predictor variables
            ls.Dist_dat  = [ls.model.p.Maps[x][ls.model.timestep, :, :] if len(ls.model.p.Maps[x].shape) == 3 else ls.model.p.Maps[x] for x in ls.Dist_vars]

            ### combine numpy arrays to single pandas       
            ls.Dist_dat  = np.array([x.reshape(ls.model.p.xlen*ls.model.p.ylen).data for x in ls.Dist_dat]).transpose()
        
            ### Parallel prediction
            boot_frame[type(ls).__name__] = make_boot_frame(ls)
    
    
    #########################################################################################
    ### check all split val updates coorespond to a split not a leaf
    #########################################################################################
    
    for ls in boot_frame.keys():
        
        tmp = boot_frame[ls]['df'][1]
        tmp = np.select([tmp['var'] != '<leaf>'], [tmp['splits.cutleft']], 'leaf')[tmp['var'] == '<leaf>']
        
        if any(tmp != 'leaf'):
            
            errors.append("split parameters updated incorrectly")
    
    #########################################################################################
    ### check all split values come from correct file
    #########################################################################################
    
    for ls in boot_frame.keys():
        
        tmp = boot_frame[ls]['df'][1]
        val = tmp.iloc[np.max(np.where(np.select(
               [tmp['var'] != '<leaf>'], [tmp['splits.cutleft']], 'leaf') != 'leaf')), 5]
        split_vals = [a.boot_Dist_pars['Thresholds'] for a in mod.ls if type(a).__name__ == ls][0]
        
        if not float(val[1:len(val)]) == split_vals[len(split_vals)-1].iloc[1, 0]:
            
            errors.append("split parameters updated incorrectly")
    
    
    #########################################################################################
    ### check all probabiliities come from correct file
    #########################################################################################
    
    for ls in boot_frame.keys():
        
        tmp = boot_frame[ls]['df'][1]
        val = tmp.iloc[np.min(np.where(np.select(
               [tmp['var'] == '<leaf>'], [tmp['yprob.TRUE']], 'split') != 'split')), :].loc['yprob.TRUE']
        leaf_vals = [a.boot_Dist_pars['Probs'] for a in mod.ls if type(a).__name__ == ls][0]
        
        if not val == leaf_vals[0].iloc[1, len(leaf_vals[0].columns)-1]:
            
            errors.append("output probabilities updated incorrectly")

    
    ########################################################################
    
    ### check boostrapped prediction
    
    ########################################################################
    
    c       = Client(n_workers=2)
    futures = []
    x       = make_boot_frame(mod.ls[5])
    p       = 'yprob.TRUE'
    
    for i in range(len(x['df'])):
        
        future = c.submit(predict_from_tree_numpy, dat = x['dd'], 
                              tree = x['df'][i], struct = x['ds'], 
                              split_vars = mod.ls[5].Dist_vars,
                               prob = p, skip_val = -1e+10, na_return = 0)
                
        futures.append(future)

    results = c.gather(futures)
    
    for i in range(len(results)):
        
        if not pd.Series(results[i]).round(6).isin(pd.concat([x['df'][i]['yprob.TRUE'].round(6), pd.Series(0)])).all():
            
            errors.append('boostrapped prediction failure')
            
    c.close()
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
