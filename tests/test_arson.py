# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:47:32 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

os.chdir((wd[0:-6] + '/src/data_import'))
exec(open("local_load_up.py").read())

os.chdir(wd)
exec(open("setup_full.py").read())

os.chdir(str(wd + '/test_data/R_outputs').replace('\\', '/'))
R_results = pd.read_csv('Arson_2002.csv')

### bespoke parameters
parameters['start_run'] = 13
parameters['end_run']   = 13
parameters['Observers'] = {'arson': arson, 
                           'background_rate': background_rate, 
                           'deforestation'  : deforestation}
parameters['escaped_fire'] = False
parameters['write_annual'] = False
parameters['reporters']    = ['Managed_fire', 'Background_ignitions', 'Arson']


mod = WHAM(parameters)
mod.record = lambda: 2*2

### setup
mod.setup()

### ignite
mod.go()

#################################################################

### tests - run code to reproduce model output

#################################################################

### tree code

x, b = 'arson', 'bool'

    
mod.Observers['arson'][0].Fire_dat['arson']['bool'] = mod.Observers['arson'][0].Fire_dat['arson']['bool'].iloc[:, 0:2]
Fire_struct = define_tree_links(mod.Observers['arson'][0].Fire_use[x][b]['pars'])

tree = predict_from_tree_fast(dat = mod.Observers['arson'][0].Fire_dat[x][b], 
                              tree = mod.Observers['arson'][0].Fire_use[x][b]['pars'], struct = Fire_struct, 
                               prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0)

     
### regression code   
errors = []
    
reg = mod.Observers['arson'][0].Fire_dat['arson']['ba'].Market_access * 2.9037 - 0.49
reg = 1/(1+np.exp(0-reg))
reg_m = np.nanmean(reg)
reg = [x if x >= 0.5 else 0 for x in reg]

combo = (reg + tree) / 2


afr_vals = []
    
### Assume Nonex doesn't commit arson
for ls in ['Cropland', 'Pasture', 'Rangeland', 'Forestry']:
        
    if 'Trans' in mod.LFS[ls].keys():
                
        afr_vals.append(mod.LFS[ls]['Trans'])
        
        ### remove agroforestry from logging contribution to arson
if 'Agroforestry' in mod.AFT_scores.keys():
            
   afr_vals.append(0 - mod.AFT_scores['Agroforestry'])
        
afr_vals = np.nansum(afr_vals, axis = 0)        
afr_vals = pd.Series([x if x > 0 else 0 for x in np.array(afr_vals).reshape(mod.p.xlen*mod.p.ylen)])
  
igs = combo * np.exp(afr_vals * 1.166 -2.184) * mod.p.Maps['Mask']  


###########################################################################################

### tests

###########################################################################################


def test_tree():
    
    errors = []
    
    if any([x not in (mod.Observers['arson'][0].Fire_use['arson']['bool']['pars']['yprob.TRUE'].tolist()) for x in (tree.tolist())]):
        
        errors.append("Errors in tree prediction values")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
    
def test_regression():

    errors = []
    
    if not all(pd.Series(reg).describe()[3:6] == 0):
        
        errors.append("Errors in regression prediction values")

    if not np.nanmax(pd.Series(reg))  <= 1:
        
        errors.append("Errors in regression prediction values")

    if not np.nanmean(reg) < reg_m:
        
        errors.append("Errors in regression prediction values")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
        
    
def test_combined_mod():   
    
    assert(np.nanmin(combo)  == 0 and np.nanmax(combo)  <= 1)
    

def test_arson_ignitions():   


    assert(np.nanmean(igs) == pytest.approx(np.nanmean(mod.Observers['arson'][0].Fire_vals), 0.1))

    
def test_arson_output():
    
    igs_r = igs[R_results.iloc[:, 0].notnull()]
    
    errors = []
    
    if not (np.nanmax(igs_r) == pytest.approx(np.nanmax(R_results.iloc[:, 0]), 0.1)):
        
        errors.append("Arson outputs don't match baseline calculations")
    
    if not (np.nanmedian(igs_r) == pytest.approx(np.nanmedian(R_results.iloc[:, 0]), 0.1)):
        
        errors.append("Arson outputs don't match baseline calculations")
        
    if not (np.nanquantile(igs_r, 0.75) == pytest.approx(np.nanquantile(R_results.iloc[:, 0], 0.75), 0.1)):
        
        errors.append("Arson outputs don't match baseline calculations")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))

