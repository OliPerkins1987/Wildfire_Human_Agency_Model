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
os.chdir(str(wd + '/test_data/R_outputs').replace('\\', '/'))
Arson_frame  = pd.read_csv('Arson_output_R.csv')
AFR_frame    = pd.read_csv('Recreation_arson.csv')
Unoc_frac    = pd.read_csv('Unoccupied.csv')

##########################################################################

### tests

##########################################################################

@pytest.mark.usefixtures("mod_pars")
def test_control_predict(mod_pars):  
    
    errors = []
    
    mod                          = WHAM(mod_pars)
    mod.p.AFT_fire               = True
    mod.p.apply_fire_constraints = True
    mod.p.escaped_fire           = True
    
    mod.setup()
    mod.p.bootstrap      = False
    mod.timestep         = 0
    
    mod.ls.get_vals()
    mod.allocate_X_axis()
    
    mod.agents.compete()
    mod.allocate_AFT()
    
    mod.agents.fire_use()
    mod.calc_BA(group_lc = False)
    
    ###############################################################
    ### calculate fire control
    ###############################################################
    
    mod.Observers['fire_control_measures'][0].control()

    
    ###################################################################################
    
    ### compare with raw calculation
    
    ###################################################################################
    
    self = mod.Observers['fire_control_measures'][0]
        
    node1 = self.Control_dat['Pre'] < 0.0454666
    node2 = self.Control_dat['Intense'] < 0.149769
    node3 = self.Control_dat['Trans'] < 0.41279
    node4 = self.Control_dat['Unoc_pre'] < 0.393417
    node5 = self.Control_dat['Pre'] < 0.19152
    node6 = self.Control_dat['Unoc_pre'] < 0.128279

    leaf1 = node1 * 0.756813
    leaf2 = (1-node1) * node2 * node3 * node4 * node5 * 0.26
    leaf3 = (1-node1) * node2 * node3 * node4 * (1-node5) * 0
    leaf4 = (1-node1) * node2 * node3 * (1-node4) * 0.407767
    leaf5 = (1-node1) * node2 * (1-node3) * 0.510373
    leaf6 = (1-node1) * (1-node2) * (node6) * 0.262626
    leaf7 = (1-node1) * (1-node2) * (1-node6) * 0.75
    
    out_probs = leaf1 + leaf2 + leaf3 + leaf4 + leaf5 + leaf6 + leaf7
    
    #################################################
    
    ### test control measures correct
    
    #################################################
    
    if not(np.nanmean(self.Control_vals['pasture'] - out_probs) == pytest.approx(0, abs = 0.00001)):
            
            errors.append("Fire control measures calculated incorrectly")


    ########################################
    
    ### test escaped rate calculation
    
    ########################################

    mod.calc_escaped_fires()
    
    Escaped_fire = {}
    base = mod.p.AFT_pars['Fire_escape']['Overall']
    
    #############################################################
    ### conduct manually for hunter gatherer
    #############################################################
    
    base_rate = 0.0110
    Escaped_fire['hg'] = mod.Managed_igs['hg'] * base_rate  
    
    control_impact =  mod.p.AFT_pars['Fire_escape']['Overall']
    
    controlled = (1-out_probs) * 0.308000 
    no_control = (out_probs) * 3.246753
    
    Escaped_hg = Escaped_fire['hg'] * np.array(
        controlled+no_control).reshape(mod.ylen, mod.xlen)
    
    ### final test
    
    if not(np.nanmax(abs(mod.Escaped_fire['hg'] - Escaped_hg)) == pytest.approx(0, abs = 0.0001)):
            
            errors.append("Escaped fires calculated incorrectly") 
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    