# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:12:17 2021

@author: Oli
"""

from dask.distributed import Client
import numpy as np
import pandas as pd
from copy import deepcopy


from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation


##################################################################

### Functions to run parallel prediction across bootstrapped trees
### functions called as an AFT method, with x = self (AFT)

##################################################################

def make_boot_frame(a):
    
    ''' creates list of tree frames from bootstrapped parameters'''
    
    boot_pred = {'df':[], 'ds': deepcopy(a.Dist_struct), 
             'dd': deepcopy(a.Dist_dat)}
 
    for i in range(a.boot_Dist_pars['Thresholds'][0].shape[0]):

        boot_pred['df'].append(deepcopy(update_pars(a.Dist_frame, a.boot_Dist_pars['Thresholds'], 
                                    a.boot_Dist_pars['Probs'], method = 'bootstrapped', 
                                    target = 'yprob.TRUE', source = 'TRUE.', boot_int = i)))

    return(boot_pred)



def make_boot_frame_AFT(a, par_set = 'none'):
    
    ''' creates list of tree frames from bootstrapped parameters
    par_set = none for sub-compete type != multiple, 
    for multiple AFT sub-competition, par_set = int'''
    
    if par_set == 'none':
    
        boot_pred = {'df':[], 'ds': deepcopy(a.AFT_struct), 
             'dd': deepcopy(a.AFT_dat)}
    
        for t in range(a.boot_AFT_pars['Thresholds'][0].shape[0]):
            
            boot_pred['df'].append(deepcopy(update_pars(a.AFT_frame, a.boot_AFT_pars['Thresholds'], 
                                    a.boot_AFT_pars['Probs'], method = 'bootstrapped', 
                                    target = type(a).__name__, source = type(a).__name__, boot_int = t)))
            
    elif par_set != 'none':
        
        boot_pred = {'df':[], 'ds': deepcopy(a.AFT_struct[par_set]), 
             'dd': deepcopy(a.AFT_dat)}
        
        for t in range(a.boot_AFT_pars[par_set]['Thresholds'][0].shape[0]):
        
            boot_pred['df'].append(deepcopy(update_pars(a.AFT_frame[par_set], a.boot_AFT_pars[par_set]['Thresholds'], 
                                    a.boot_AFT_pars[par_set]['Probs'], method = 'bootstrapped', 
                                    target = type(a).__name__, source = type(a).__name__, boot_int = t)))

    return(boot_pred)


##########################################

### Uses boot_pred object to run parallel prediction

########################################## 


def parallel_predict(x, c, p):
    
    '''run a parallel prediction'''
    
    futures = []
    
    for i in range(len(x['df'])):
        
        future = c.submit(predict_from_tree_fast, dat = x['dd'], 
                              tree = x['df'][i], struct = x['ds'], 
                               prob = p, skip_val = -3.3999999521443642e+38, na_return = 0)
        
        
        futures.append(future)

    results = c.gather(futures)
    
    return(results)


def combine_bootstrap(a):
    
    ''' Combine parallel prediction outputs'''
    
    ### Apply zeroing out
    dv = [0 if x<= a.p.theta else x for y in a.Dist_vals for x in y]           
    dv = np.array(dv).reshape(a.boot_Dist_pars['Thresholds'][0].shape[0], 
                              a.p.ylen, a.p.xlen)
            
    ### Combine
    dv = pd.Series(np.nanmean(dv, axis = 0).reshape(a.model.p.ylen*a.model.p.xlen)).to_list()
    
    return(dv)
  