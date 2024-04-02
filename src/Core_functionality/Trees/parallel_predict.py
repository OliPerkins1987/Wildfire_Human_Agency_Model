# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:12:17 2021

@author: Oli
"""

from dask.distributed import Client
import numpy as np
import pandas as pd
from copy import deepcopy


from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast
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

##########################################

### Uses boot_pred object to run parallel prediction

########################################## 


def parallel_predict(x, c, p):
    
    '''run a parallel prediction'''
    
    futures = []
    
    for i in range(len(x['df'])):
        
        future = c.submit(predict_from_tree_fast, dat = x['dd'], 
                              tree = x['df'][i], struct = x['ds'], 
                               prob = p, skip_val = -1e+10, na_return = 0)
        
        
        futures.append(future)

    results = c.gather(futures)
    
    return(results)


def combine_bootstrap(a):
    
    ''' Combine parallel prediction outputs'''
    
    dv = a.Dist_vals
    
    ### Combine
    dv = pd.Series(np.nanmean(dv, axis = 0))
    
    ### Apply zeroing out
    dv = dv.where(dv < 0.1, 0).to_list()
    
    return(dv)
  