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
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap


#########################################################################

### Load test data

#########################################################################

os.chdir(r'C:/Users/Oli/Documents/PIES/WHAMv2/mod/tests/test_data/R_outputs')
Forestry_test   = pd.read_csv('Forestry_dist.csv')
Unoccupied_test = pd.read_csv('Unoccupied_dist.csv')

##########################################################################

### tests

##########################################################################

@pytest.mark.usefixtures("mod_pars")
def test_ls_allocation(mod_pars):  
    
    errors = []
    
    ### setup model
    mod                  = WHAM(mod_pars)
    mod.p.bootstrap      = False
    mod.p.numb_bootstrap = 2
    mod.timestep         = 24
    mod.xlen             = mod_pars['xlen']
    mod.ylen             = mod_pars['ylen']
    
    # Create grid
    mod.grid = ap.Grid(mod, (mod.xlen, mod.ylen), track_empty=False)
    mod.Area = np.array(mod.p.Maps['Area']).reshape(mod.p.ylen, mod.p.xlen)
        
    # Create land systems
    mod.ls     = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.LS]])
    
    ### setup land systems
    mod.ls.setup()
    mod.ls.get_pars(mod.p.AFT_pars)
    mod.ls.get_boot_vals(mod.p.AFT_pars)
      
    
    ##########################################################
    ### are the land system competitions executed correctly?
    ##########################################################
        
    mod.ls.get_vals()
    
    ls_scores    = {'Forestry':mod.ls[3].Dist_vals, 'Unoccupied':mod.ls[5].Dist_vals}
    
    ### Forestry predictions
    
    Forestry_filt= (np.isnan(ls_scores['Forestry']) == False) & np.isnan(Forestry_test.iloc[:, 0] == False)
    
    if not ls_scores['Forestry'][Forestry_filt] == pytest.approx(
            np.array(Forestry_test[Forestry_filt].iloc[:, 0])):
        
        errors.append("Forestry distribution predicted incorrectly")
    
    ### Unoccupied predictions
    
    Unoc_filt= (np.isnan(ls_scores['Unoccupied']) == False) & (np.isnan(Unoccupied_test.iloc[:, 0]) == False)
    
    if not ls_scores['Unoccupied'][Unoc_filt] == pytest.approx(
            np.array(Unoccupied_test[Unoc_filt].iloc[:, 0])):
        
        errors.append("Unoccupied distribution predicted incorrectly")
    
    ##############################################################
    
    ### test x axis allocation
    
    ##############################################################
    
    mod.allocate_X_axis()
    
    alloc_frame = sum(mod.X_axis.values()).reshape(mod.xlen*mod.ylen)
    alloc_filt  = (np.isnan(alloc_frame) == False) & (np.isnan(mod.p.Maps['Mask']) == False)
    
    if not alloc_frame[alloc_filt] == pytest.approx(mod.p.Maps['Mask'][alloc_filt]):
        
        errors.append("Unoccupied distribution predicted incorrectly")
    
    
    
    
    
    
