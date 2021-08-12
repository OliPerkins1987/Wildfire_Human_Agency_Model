# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np
from copy import deepcopy

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast


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
            
            if self.sub_AFT['kind'] != 'Multiple':
            
                self.AFT_frame  = AFT_dict['AFT_dist'][str('Sub_AFTs' + '/' + self.sub_AFT['afr'] + '_' + self.sub_AFT['ls'])]
                self.AFT_struct = define_tree_links(self.AFT_frame)
                self.AFT_vars   = [x for x in self.AFT_frame.iloc[:,1].tolist() if x != '<leaf>']
            
            else:
                
                ### Where AFT splits across more than 2 LFS
                ### afr & LFS should be lists of same length
                
                self.AFT_frame  = []
                self.AFT_struct = []
                self.AFT_vars   = []
                
                for i in range(len(self.sub_AFT['afr'])):
                    
                    self.AFT_frame.append(AFT_dict['AFT_dist'][str('Sub_AFTs' + '/' + self.sub_AFT['afr'][i] + '_' + self.sub_AFT['ls'][i])])
                    self.AFT_struct.append(define_tree_links(self.AFT_frame[i]))
                    self.AFT_vars.append([x for x in self.AFT_frame[i].iloc[:,1].tolist() if x != '<leaf>'])
                
            
        else:
            
            self.AFT_frame  = 'None'
            
        
       
        
    def get_boot_vals(self, AFT_dict):
            
            ### get parameter values for bootstrapping of tree thresholds
            
            self.boot_Dist_pars   = {}
            self.boot_AFT_pars    = {}
            
            if self.p.bootstrap == True:
                

                self.boot_Dist_pars['Thresholds']           = AFT_dict['Dist_pars']['Thresholds'][str(self.ls + '/' + self.afr)]
                self.boot_Dist_pars['Probs']                = AFT_dict['Dist_pars']['Probs'][str(self.ls + '/' + self.afr)]
        
        
                if self.sub_AFT['exists'] == True:
            
                    if self.sub_AFT['kind'] != 'Multiple':
            
                        self.boot_AFT_pars['Thresholds']           = AFT_dict['Dist_pars']['Thresholds'][str('Sub_AFTs' + '/' + self.sub_AFT['afr'] + '_' + self.sub_AFT['ls'])]
                        self.boot_AFT_pars['Probs']                = AFT_dict['Dist_pars']['Probs'][str('Sub_AFTs' + '/' + self.sub_AFT['afr'] + '_' + self.sub_AFT['ls'])]
    
        
                    else:
                
                        ### Where AFT splits across more than 2 LFS
                        ### afr & LFS should be lists of same length
                
                        self.boot_AFT_pars  = []
                
                        for i in range(len(self.sub_AFT['afr'])):
                    
                            self.boot_AFT_pars.append({})        
                    
                            self.boot_AFT_pars[i]['Thresholds']           = AFT_dict['Dist_pars']['Thresholds'][str('Sub_AFTs' + '/' + self.sub_AFT['afr'][i] + '_' + self.sub_AFT['ls'][i])]
                            self.boot_AFT_pars[i]['Probs']                = AFT_dict['Dist_pars']['Probs'][str('Sub_AFTs' + '/' + self.sub_AFT['afr'][i] + '_' + self.sub_AFT['ls'][i])]
          
            
                else:
            
                    self.boot_AFT_pars  = 'None'
        
        
        
        
        
        ### Add Fire use
        
    
    #########################################################################

    ### AFT Distribution

    #########################################################################    
    
    def compete(self):
        
        ''' 
        Competition between LFS
        
        Can we find a way to stop predicting over duplicate parameter sets for LFS?
        '''
        
            ### single set of parameter values
        if self.p.bootstrap != True:
        
            ### gather correct numpy arrays 4 predictor variables
            self.Dist_dat  = [self.model.p.Maps[x][self.model.p.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]


            ### combine numpy arrays to single pandas       
            self.Dist_dat  = pd.DataFrame.from_dict(dict(zip(self.Dist_vars, 
                           [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat])))
        
            ### do prediction
            self.Dist_vals = predict_from_tree_fast(dat = self.Dist_dat, 
                              tree = self.Dist_frame, struct = self.Dist_struct, 
                               prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0)
        
        
        
            ### apply theta zero-ing out constraint
            self.Dist_vals = [0 if x <= self.p.theta else x for x in self.Dist_vals]
            
            
            ### bootstrapped version
        elif self.p.bootstrap == True:
            
            self.Dist_vals = []
            
            ### gather correct numpy arrays 4 predictor variables
            self.Dist_dat  = [self.model.p.Maps[x][self.model.p.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]

            ### combine numpy arrays to single pandas       
            self.Dist_dat  = pd.DataFrame.from_dict(dict(zip(self.Dist_vars, 
                              [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat])))
        
            
            for i in range(self.boot_Dist_pars['Thresholds'][0].shape[0]):
                
                self.Dist_frame = update_pars(self.Dist_frame, self.boot_Dist_pars['Thresholds'], 
                                    self.boot_Dist_pars['Probs'], method = 'bootstrapped', 
                                    target = 'yprob.TRUE', source = 'TRUE.', boot_int = i)
           
                ### do prediction
                d         = deepcopy(self.Dist_dat)
                
                Dist_vals = predict_from_tree_fast(dat = d, 
                              tree = self.Dist_frame, struct = self.Dist_struct, 
                               prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0)
        
                ### apply theta zero-ing out constraint
                self.Dist_vals.append([0 if x <= self.p.theta else x for x in Dist_vals])
            
            self.Dist_vals = pd.DataFrame(np.column_stack(self.Dist_vals)).mean(axis = 1).to_list()
            
        
    def sub_compete(self):
        
        ''' Competition between AFTs within each LFS '''
        
        ### 1 parameter version
        
        if self.sub_AFT['exists'] == True and self.p.bootstrap == False:
        
            if self.sub_AFT['kind'] != 'Multiple':    
        
                ### gather correct numpy arrays 4 predictor variables
                self.AFT_dat   = [self.model.p.Maps[x][self.model.p.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.AFT_vars]
        
                ### combine numpy arrays to single pandas       
                self.AFT_dat   = pd.DataFrame.from_dict(dict(zip(self.AFT_vars, 
                                  [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat])))
            
                ### do prediction
                self.AFT_vals  = predict_from_tree_fast(self.AFT_dat, tree = self.AFT_frame, 
                                 struct = self.AFT_struct, prob = type(self).__name__, 
                                  skip_val = -3.3999999521443642e+38, na_return = 0)

            
            elif self.sub_AFT['kind'] == 'Multiple':
                
                self.AFT_dat  = []
                self.AFT_vals = []
                
                for i in range(len(self.sub_AFT['afr'])): 
            
                    ### gather correct numpy arrays 4 predictor variables
                    self.AFT_dat.append([self.model.p.Maps[x][self.model.p.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.AFT_vars[i]])
        
                    ### combine numpy arrays to single pandas       
                    self.AFT_dat[i]   = pd.DataFrame.from_dict(dict(zip(self.AFT_vars[i], 
                                         [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat[i]])))
            
                    ### do prediction - these are added together in the WHAM AFT allocate routine
                    self.AFT_vals.append(predict_from_tree_fast(self.AFT_dat[i], tree = self.AFT_frame[i], 
                                 struct = self.AFT_struct[i], prob = type(self).__name__, 
                                  skip_val = -3.3999999521443642e+38, na_return = 0))

        
        
        ### bootstrapped parameters
        
        elif self.sub_AFT['exists'] == True and self.p.bootstrap == True:
        
            if self.sub_AFT['kind'] != 'Multiple':    
        
                ### gather correct numpy arrays 4 predictor variables
                self.AFT_dat   = [self.model.p.Maps[x][self.model.p.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.AFT_vars]
        
                ### combine numpy arrays to single pandas       
                self.AFT_dat   = pd.DataFrame.from_dict(dict(zip(self.AFT_vars, 
                                  [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat])))
            
            
                ### run through bootstrapped parameter values
                self.AFT_vals  = []
            
                for i in range(self.boot_AFT_pars['Thresholds'][0].shape[0]):
                    
                    ### update tree frame with bootstrapped parameters
                    self.AFT_frame = update_pars(self.AFT_frame, self.boot_AFT_pars['Thresholds'], 
                                    self.boot_AFT_pars['Probs'], method = 'bootstrapped', 
                                    target = type(self).__name__, source = type(self).__name__, boot_int = i)
           
                    ### do prediction
                    a = deepcopy(self.AFT_dat)
                    
                    self.AFT_vals.append(predict_from_tree_fast(a, tree = self.AFT_frame, 
                                 struct = self.AFT_struct, prob = type(self).__name__, 
                                  skip_val = -3.3999999521443642e+38, na_return = 0))
                                     
            
                self.AFT_vals = pd.DataFrame(np.column_stack(self.AFT_vals)).mean(axis = 1).to_list()
        
        
            elif self.sub_AFT['kind'] == 'Multiple':
        
                self.AFT_dat  = []
                self.AFT_vals = []
                
                for i in range(len(self.sub_AFT['afr'])): 
            
                    self.AFT_vals.append([])
                                    
                    ### gather correct numpy arrays 4 predictor variables
                    self.AFT_dat.append([self.model.p.Maps[x][self.model.p.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.AFT_vars[i]])
        
                    ### combine numpy arrays to single pandas       
                    self.AFT_dat[i]   = pd.DataFrame.from_dict(dict(zip(self.AFT_vars[i], 
                                         [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat[i]])))
            
                    
                    for j in range(self.boot_AFT_pars[i]['Thresholds'][0].shape[0]):
                
                        ### update tree frame with bootstrapped parameters        
                        self.AFT_frame[i] = update_pars(self.AFT_frame[i], self.boot_AFT_pars[i]['Thresholds'], 
                                    self.boot_AFT_pars[i]['Probs'], method = 'bootstrapped', 
                                    target = type(self).__name__, source = type(self).__name__, boot_int = j)
           
                        ### do prediction
                        a = deepcopy(self.AFT_dat[i])
                        
                        self.AFT_vals[i].append(predict_from_tree_fast(a, tree = self.AFT_frame[i], 
                                 struct = self.AFT_struct[i], prob = type(self).__name__, 
                                  skip_val = -3.3999999521443642e+38, na_return = 0))

            
                    self.AFT_vals[i] = pd.DataFrame(np.column_stack(self.AFT_vals[i])).mean(axis = 1).to_list()
            
        else:
            
            self.AFT_vals  = 'None'
    
    
    #######################################################################
    
    ### Fire
    
    #######################################################################
    
    
    def fire_use(self):
        
        pass
    
    
    def fire_suppression(self):
        
        pass

##################################################################

### dummy agents for testing

##################################################################

class dummy_agent(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Test'
        self.ls  = 'Test'
        
        self.sub_AFT = {'exists': True, 'kind': 'Addition',  
                        'afr': 'Test', 'ls': 'Test'}    


class multiple_agent(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Test'
        self.ls  = 'Test'
        
        self.sub_AFT = {'exists': True, 'kind': 'Multiple',  
                        'afr': ['Test', 'Test'], 'ls': ['Test', 'Test']}    
        
        
        
        