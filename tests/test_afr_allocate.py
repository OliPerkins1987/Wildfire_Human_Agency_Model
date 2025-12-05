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


##########################################################################

### tests

##########################################################################

@pytest.mark.usefixtures("mod_pars")
def test_afr_allocate(mod_pars):  
    
    errors = []
    
    ### setup model
    mod                  = WHAM(mod_pars)
    mod.p.bootstrap      = False
    mod.p.numb_bootstrap = 2
    mod.timestep         = 0
    mod.xlen             = mod_pars['xlen']
    mod.ylen             = mod_pars['ylen']
    
    ### make ls list
    mod.ls     = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.LS]])
    
    ### setup ls
    mod.ls.setup()
    mod.ls.get_pars(mod.p.AFT_pars)
    
    ### make agent list
    mod.agents           = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.AFTs]])
    
    ### setup agents
    mod.agents.setup()
    mod.agents.get_dist_pars(mod.p.AFT_pars)
    
    ### run aft allocation 
    mod.ls.get_vals()
    mod.allocate_X_axis()
    mod.agents.compete()
    mod.allocate_AFT()
    
    
    #############################
    
    ### test allocation correct
    
    #############################
    
    Pre = (mod.AFT_scores['Swidden'] + mod.AFT_scores['Pastoralist_r'] + mod.AFT_scores['Pastoralist_p'] +
           mod.AFT_scores['Hunter_gatherer_f'] + mod.AFT_scores['Hunter_gatherer_n'])

    if not(np.nanmax(Pre - mod.AFR['Pre']) == pytest.approx(0, abs=1e-07)):
            
            errors.append("AFR calculated incorrectly")
            
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
