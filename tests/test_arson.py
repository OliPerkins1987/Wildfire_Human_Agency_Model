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
def test_arson_predict(mod_pars):  
    
    errors = []
    
    mod                  = WHAM(mod_pars)
    mod.setup()
    mod.p.bootstrap      = False
    mod.timestep         = 0
    
    mod.ls.get_vals()
    mod.allocate_X_axis()
    
    ###############################################################
    ### calculate arson with prescribed AFTs
    ###############################################################
    
    self     = mod.Observers['arson'][0]
    
    fire_hab = (self.model.p.Maps['ET'].data[self.model.timestep, :, :] > 641)
    
    afr_vals = AFR_frame['Recreationalist'].values.reshape(self.model.ylen, self.model.xlen)
    
    ### Simple representation of land conflict
    self.Fire_vals = (1/(1+np.exp(
            0-(-1.8968080 + (afr_vals*21.8860313) -(0.0009795 * self.model.p.Maps['Market.Inf'].data[self.model.timestep, :, :]))))) 
        
    ### account for higher HDI regions
    self.Fire_vals = self.Fire_vals * (
            1/(1+np.exp(0-(3.999-10.416*self.model.p.Maps['HDI'].data[self.model.timestep, :, :])))) 
        
    ### adjust for land area of pixel & ecological limits
    self.Fire_vals = self.Fire_vals * fire_hab
    self.Fire_vals = self.Fire_vals * self.model.p.Maps['Mask'].reshape(self.model.ylen, self.model.xlen)
        
    
    ###################################################################################
    
    ### compare with R calculation
    
    ###################################################################################
    
    Arson_frac = Arson_frame['Arson_frac'].values.reshape(mod.ylen, mod.xlen)
    
    #############################
    
    ### test allocation correct
    
    #############################
    
    if not(np.nanmean(self.Fire_vals - Arson_frac) == pytest.approx(0, abs = 0.001)):
            
            errors.append("Arson calculated incorrectly")

    if not(np.nanmedian(self.Fire_vals - Arson_frac) == pytest.approx(0, abs = 0.0001)):
            
            errors.append("Arson calculated incorrectly")


    ########################################
    
    ### test arson constraint
    
    ########################################
    
    for c in self.model.Observers.values():
            
            if 'ct' in type(c[0]).__name__:
                
                c.constrain_arson()

    self.Fire_vals = Arson_frac * (1-Unoc_frac['Unoc_frac'].values.reshape(mod.ylen, mod.xlen))
    
    if not(np.nanmax((self.Fire_vals - Arson_frac)) == pytest.approx(0, abs = 0.0001)):
            
            errors.append("Arson constraint incorrect") 
    
    if not(np.nanmean((self.Fire_vals - Arson_frac)) == pytest.approx(0, abs = 0.005)):
            
            errors.append("Arson constraint incorrectly")
            
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    