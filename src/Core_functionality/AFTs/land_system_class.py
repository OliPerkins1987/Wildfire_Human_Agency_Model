# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast, predict_from_tree_numpy
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap

from copy import deepcopy

class land_system(ap.Agent):
    
    ''' 
    Model class to hold information on land system distribution
    '''
    
    def setup(self):
        
        '''
        
        Basic constants:
        dist_method - how does this land system distribute itself: 
        through competition or through prescribed inputs?
                    
        'specified' allows distribution method to be defined at the LS level itself
        
        '''  
      
        self.dist_method = 'Prescribed' #or 'Competition' or 'Specified'
        self.pars_key    = '' ## what is the key in the core_pars dictionary where parameterisation is stored 
        
        
    def get_pars(self, LS_dict):
        
        
        ### Get distribution parameters
        if self.dist_method == 'Competition':
            
            self.Dist_frame  = LS_dict['AFT_dist'][self.pars_key]
            self.Dist_struct = define_tree_links(self.Dist_frame)
            self.Dist_vars   = [x for x in self.Dist_frame.iloc[:,1].tolist() if x != '<leaf>']
            
        elif self.dist_method == 'Prescribed':
            
            self.Dist_frame  = 'None'
            
        elif self.dist_method == 'Specified':
        
            pass
        
        
    def get_boot_vals(self, LS_dict):
        
        if self.dist_method == 'Competition':
        
            self.boot_Dist_pars   = {'Thresholds':'', 
                                       'Probs': ''}    
        
            self.boot_Dist_pars['Thresholds']   = LS_dict['Dist_pars']['Thresholds'][self.pars_key]
            self.boot_Dist_pars['Probs']        = LS_dict['Dist_pars']['Probs'][self.pars_key]
        
            ### filter number of bootstraps used
            if self.p.numb_bootstrap != 'max':
            
                for i in range(len(self.boot_Dist_pars['Thresholds'])):
                
                    self.boot_Dist_pars['Thresholds'][i] = self.boot_Dist_pars['Thresholds'][i].iloc[0:self.p.numb_bootstrap, :]
            
                for i in range(len(self.boot_Dist_pars['Probs'])):
                
                    self.boot_Dist_pars['Probs'][i] = self.boot_Dist_pars['Probs'][i].iloc[0:self.p.numb_bootstrap, :]
        
        elif self.dist_method == 'Prescribed':
            
            self.boot_Dist_pars = 'None'
            
        elif self.dist_method == 'Specified':
        
            pass
    
    
    #########################################################################

    ### AFT Distribution

    #########################################################################    
    
    def get_vals(self):
        
        ''' 
        Raw fraction / competition scores for land systems
        Feeds into Allocate_X_axis in WHAM class
        
        '''
        
        if self.dist_method == 'Competition' and self.model.p.bootstrap == False:
                
            
            ### gather correct numpy arrays 4 predictor variables
            self.Dist_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]
            
            self.Dist_dat  = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat]).transpose()
        
            ### do prediction - NB zeroing out not applied for ls
            self.Dist_vals = np.array(predict_from_tree_numpy(dat = self.Dist_dat, 
                              tree = self.Dist_frame, split_vars = self.Dist_vars, struct = self.Dist_struct,
                               prob = 'yprob.TRUE', skip_val = -1e+10, na_return = 0))
            

        elif self.dist_method == 'Competition' and self.model.p.bootstrap == True:
            
            self.Dist_vals = []
            
            ### gather numpy arrays of predictor variables
            self.Dist_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]

            ### combine numpy arrays to single pandas       
            self.Dist_dat  = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat]).transpose()
        
            ### Parallel prediction - no zeroing out for ls
            boot_frame     = make_boot_frame(self)
            dv             = parallel_predict(boot_frame, self.model.client, 'yprob.TRUE', self.Dist_vars)
            self.Dist_vals = np.nanmean(dv, axis = 0)  
            
            
        elif self.dist_method == 'Prescribed':
        
            ### NB uses agent class name to identify map - second line removes missing
            self.Dist_vals  = self.model.p.Maps[type(self).__name__][self.model.timestep, :, :].reshape(self.model.p.xlen*self.model.p.ylen).data
            self.Dist_vals  = np.select([self.Dist_vals>0], [self.Dist_vals], default = 0)
        
        
        elif self.dist_method == 'Specified':
            
            ### method for establishing competitiveness score specified in individual ls sub_class            
            pass

  

