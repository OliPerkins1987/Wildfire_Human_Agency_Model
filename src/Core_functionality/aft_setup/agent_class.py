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
        fc = fractional coverage of AFT's grid cell
            
        '''  
      
        self.ls = ''
        self.afr= ''
        self.fc = 0
        
    def get_pars(self, AFT_dict):
        
        ### Distribution tree
        
        self.Dist_frame = AFT_dict['AFT_dist'][str(self.ls + '/' + self.afr)]
        self.Dist_struct= define_tree_links(self.Dist_frame)
        self.Dist_pars  = [x for x in self.Dist_frame.iloc[:,1].tolist() if x != '<leaf>']
        
        ### Dist parameter vals - needs to include vals for bootstrapping
        
        
        ### Fire use
        
        
    def compete(self):
        
        ''' 
        Competition between AFTs - 
        currently only works for a single set of parameter vals
        
        '''
        
        ### gather correct numpy arrays 4 predictor variables
        self.Dist_dat = dict(zip(self.Dist_pars, 
                         [self.model.p.Maps[x][self.model.p.timestep, :, :] for x in self.Dist_pars]))

        ''' Should this code be factored out into a separate function??'''
        ### convert numpy arrays to pandas
        shp           = [x for x in self.Dist_dat.values()][0].shape
        
        self.Dist_dat = pd.DataFrame(np.column_stack([self.Dist_dat[x].reshape(shp[0]*shp[1], 
                            1) for x in self.Dist_dat.keys()]))
        
        self.Dist_dat.columns = self.Dist_pars
        
        ### do prediction
        self.Dist_vals = self.Dist_dat.apply(predict_from_tree, 
                          axis = 1, tree = self.Dist_frame, struct = self.Dist_struct, 
                           prob = 'yprob.TRUE')
        
        self.Dist_vals = np.array(self.Dist_vals).reshape(shp[1], shp[0])
        
        ### push competition score to some kind of central parameter array
        
        pass
    
    
    def fire_use(self):
        
        pass
    
    
    def fire_suppression(self):
        
        pass


