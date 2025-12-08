# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap

from copy import deepcopy

###########################################################################################

### Prescribed input LS

###########################################################################################

class Cropland(land_system):

    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Prescribed'


class Pasture(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Prescribed'
        
        
class Rangeland(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Prescribed'


class Urban(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Prescribed'



#################################################################################

### LS distributed through competition 

#################################################################################

class Unoccupied(land_system):
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Competition'
        self.pars_key    = 'Xaxis/Unoccupied'



#################################################################################

### LS distributed through other specification

#################################################################################


class Forestry(land_system):
    
    ### Essentially prescribed - but Forest cover != Forestry fraction
    ### See sub-routine in wham (allocate X-axis)
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Specified'
        
    def get_vals(self):
        self.Dist_vals  = self.model.p.Maps['Forest'][self.model.timestep, :, :].reshape(self.model.p.xlen*self.model.p.ylen).data
        self.Dist_vals  = np.array([x if x >= 0 else 0 for x in self.Dist_vals])



class Nonex(land_system):
    
    ### Has two sets of parameters - one to compete against Forestry
    ### & the other to compete for other vegetation (grass & shrubs)
    
    def setup(self):
        land_system.setup(self)
        self.dist_method = 'Specified'
        self.pars_key    = {'Forest': 'Xaxis/Forest', 'Other': 'Xaxis/Other'}
    
    def get_pars(self, LS_dict):
              
        ### two sets of pars
        self.Dist_frame  = {'Forest': LS_dict['AFT_dist'][self.pars_key['Forest']]}
        self.Dist_struct = {'Forest': define_tree_links(self.Dist_frame['Forest'])}
        self.Dist_vars   = {'Forest': [x for x in self.Dist_frame['Forest'].iloc[:,1].tolist() if x != '<leaf>']}
            
        self.Dist_frame['Other']   = LS_dict['AFT_dist'][self.pars_key['Other']]
        self.Dist_struct['Other']  = define_tree_links(self.Dist_frame['Other'])
        self.Dist_vars['Other']    = [x for x in self.Dist_frame['Other'].iloc[:,1].tolist() if x != '<leaf>']   
    
    
    def get_boot_vals(self, LS_dict):
    
            self.boot_Dist_pars   = {'Forest': {'Thresholds':'', 
                                       'Probs': ''}, 
                                     'Other': {'Thresholds':'', 
                                       'Probs': ''}}    
        
            ### two sets of pars
            self.boot_Dist_pars['Forest']['Thresholds']   = LS_dict['Dist_pars']['Thresholds'][self.pars_key['Forest']]
            self.boot_Dist_pars['Forest']['Probs']        = LS_dict['Dist_pars']['Probs'][self.pars_key['Forest']]  
        
            self.boot_Dist_pars['Other']['Thresholds']   = LS_dict['Dist_pars']['Thresholds'][self.pars_key['Other']]
            self.boot_Dist_pars['Other']['Probs']        = LS_dict['Dist_pars']['Probs'][self.pars_key['Other']]    
    
    
    def get_vals(self):
        
        self.Dist_vals = {}
        
        ### single parameter version
        if self.model.p.bootstrap == False:
        
            for k in self.pars_key.keys():  
        
                self.Dist_dat     = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars[k]]


                ### combine numpy arrays to single pandas       
                self.Dist_dat     = pd.DataFrame.from_dict(dict(zip(self.Dist_vars[k], 
                                 [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat])))
        
                ### do prediction - Theta not applied to ls classes
                self.Dist_vals[k] = np.array(predict_from_tree_fast(dat = self.Dist_dat, 
                                     tree = self.Dist_frame[k], struct = self.Dist_struct[k], 
                                      prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0))


        ### bootstrapped version
        
        if self.model.p.bootstrap == True:
        
            for k in self.pars_key.keys():
            
                self.Dist_vals[k] = []            
        
                self.Dist_dat     = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars[k]]


                ### combine numpy arrays to single pandas       
                self.Dist_dat  = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat]).transpose()
        
        
                ### Parallel prediction
                boot_pred = {'df':[], 'ds': deepcopy(self.Dist_struct[k]), 
                    'dd': deepcopy(self.Dist_dat)}
 
                for i in range(self.boot_Dist_pars[k]['Thresholds'][0].shape[0]):

                    boot_pred['df'].append(deepcopy(update_pars(self.Dist_frame[k], self.boot_Dist_pars[k]['Thresholds'], 
                                    self.boot_Dist_pars[k]['Probs'], method = 'bootstrapped', 
                                    target = 'yprob.TRUE', source = 'TRUE.', boot_int = i)))
                
                ### Gather & Combine, NB: Theta not applied to ls classes
                dv                = parallel_predict(boot_pred, self.model.client, 'yprob.TRUE', self.Dist_vars[k])
                self.Dist_vals[k] = np.nanmean(dv, axis = 0)
                
                
        
