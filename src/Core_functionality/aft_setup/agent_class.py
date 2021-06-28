# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree


class AFT(ap.Agent):
    
    ''' 
    Core model class containing key drivers of model function
    '''
    
    def setup(self):
        
        '''
        
        Basic constants:
        ls = land system of AFT
        afr= anthropogenic fire regime of AFT
        sub_AFT = does this AFT subdivide an LFS? Kind is one of fraction, addition
                    
        '''  
      
        self.ls = ''
        self.afr= ''
        self.sub_AFT = {'exists': False,
                        'kind'  : ''}
                
    def get_pars(self, AFT_dict):
        
        ### Distribution tree for LFS
        self.Dist_frame  = AFT_dict['AFT_dist'][str(self.ls + '/' + self.afr)]
        self.Dist_struct = define_tree_links(self.Dist_frame)
        self.Dist_vars   = [x for x in self.Dist_frame.iloc[:,1].tolist() if x != '<leaf>']
                
        
        ### Sub-split from LFS to AFT
        if self.sub_AFT['exists'] == True:
            
            self.AFT_frame = AFT_dict['AFT_dist'][str('Sub_AFTs' + '/' + self.sub_AFT['afr'] + '_' + self.sub_AFT['ls'])]
            self.AFT_struct= define_tree_links(self.AFT_frame)
            self.AFT_vars   = [x for x in self.AFT_frame.iloc[:,1].tolist() if x != '<leaf>']
            
        else:
            
            self.AFT_frame = 'None'
        
        
        ### Add parameter vals - needs to include vals for bootstrapping
        
        
        ### Add Fire use
        
    
    #########################################################################

    ### AFT Distribution

    #########################################################################    
    
    def compete(self):
        
        ''' 
        Competition between LFS - currently only works for a single set of parameter vals
        
        Can we find a way to stop predicting over duplicate parameter sets for LFS?
        '''
        
        ### gather correct numpy arrays 4 predictor variables
        self.Dist_dat  = [self.model.p.Maps[x][self.model.p.timestep, :, :] for x in self.Dist_vars]


        ### combine numpy arrays to single pandas       
        self.Dist_dat  = pd.DataFrame.from_dict(dict(zip(self.Dist_vars, 
                           [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat])))
        
        ### do prediction
        self.Dist_vals = self.Dist_dat.apply(predict_from_tree, 
                          axis = 1, tree = self.Dist_frame, struct = self.Dist_struct, 
                           prob = 'yprob.TRUE')
        
        ### apply theta zero-ing out constraint
        self.Dist_vals = [0 if x <= self.p.theta else x for x in self.Dist_vals]
           
        
    def sub_compete(self):
        
        ''' Competition between AFTs within each LFS '''
        
        if self.sub_AFT['exists'] == True:
        
            ### gather correct numpy arrays 4 predictor variables
            self.AFT_dat   = [self.model.p.Maps[x][self.model.p.timestep, :, :] for x in self.AFT_vars]
        
            ### combine numpy arrays to single pandas       
            self.AFT_dat   = pd.DataFrame.from_dict(dict(zip(self.AFT_vars, 
                              [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat])))
            
            ### do prediction
            self.AFT_vals  = self.AFT_dat.apply(predict_from_tree, 
                              axis = 1, tree = self.AFT_frame, struct = self.AFT_struct, 
                               prob = type(self).__name__)
            
            
        else:
            
            self.AFT_vals  = 'None'
    
    
    #######################################################################
    
    ### Fire
    
    #######################################################################
    
    
    def fire_use(self):
        
        pass
    
    
    def fire_suppression(self):
        
        pass


