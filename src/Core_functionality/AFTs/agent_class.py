
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
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.Trees.parallel_predict import make_boot_frame, make_boot_frame_AFT, parallel_predict, combine_bootstrap


class AFT(ap.Agent):
    
    ''' 
    Core model class containing key drivers of model function
    '''
    
    def setup(self):
        
        '''
        
        Basic constants:
        ls = land system of AFT
        afr= anthropogenic fire regime of AFT
                    
        '''  
      
        self.ls = ''
        
        self.afr= ''        
        self.Fire_use = {}
        self.Fire_vars = {}
        
        
    def get_dist_pars(self, AFT_dict):
        
        file_key         = str(self.ls + '/' + type(self).__name__)
        
        ### Distribution tree for LFS
        self.Dist_frame  = AFT_dict['AFT_dist'][file_key]
        self.Dist_struct = define_tree_links(self.Dist_frame)
        self.Dist_vars   = [x for x in self.Dist_frame.iloc[:,1].tolist() if x != '<leaf>']
        
        if self.p.bootstrap == True:
            
            self.boot_Dist_pars   = {}
            self.boot_AFT_pars    = {}
                
            self.boot_Dist_pars['Thresholds']  = AFT_dict['Dist_pars']['Thresholds'][file_key]
            self.boot_Dist_pars['Probs']       = AFT_dict['Dist_pars']['Probs'][file_key]
               
        
    ### Fire use parameters
    def get_fire_pars(self):
               
        
        for par in self.Fire_use.keys():
        
        ### get parameters for fire use bool (yes/no)    
        
            #parameters can be specified in par dict directly either with a pandas
            # {'type': 'constant', 'pars':float} for a constant value
            if self.Fire_use[par]['bool'] in ['lin_mod', 'tree_mod']: 
        
                self.Fire_use[par]['bool'] = {'type': self.Fire_use[par]['bool'], 
                                                'pars': self.p.AFT_pars['Fire_use']['bool'][par + '/' + type(self).__name__]}
            
                self.Fire_vars[par] = {}
                
                ###########################################
                ### extract parameter names
                ###########################################
                
                if self.Fire_use[par]['bool']['type'] == 'lin_mod':
                
                    self.Fire_vars[par]['bool'] = [x for x in self.Fire_use[par]['bool']['pars'].iloc[:,0].tolist() if x != 'Intercept']      
                
                elif self.Fire_use[par]['bool']['type'] == 'tree_mod':
                    
                    self.Fire_vars[par]['bool'] = [x for x in self.Fire_use[par]['bool']['pars'].iloc[:,1].tolist() if x != '<leaf>']
            
        
            ### get parameters for fire use degree (target % burned area)
        
            if self.Fire_use[par]['ba'] in ['lin_mod', 'tree_mod']: 
            
                self.Fire_use[par]['ba']   = {'type': self.Fire_use[par]['ba'], 
                                              'pars': self.p.AFT_pars['Fire_use']['ba'][par + '/' + type(self).__name__]}
    
                self.Fire_vars[par] = {} if not par in self.Fire_vars.keys() else self.Fire_vars[par]
    
    
                ###########################################
                ### extract parameter names
                ###########################################
                if self.Fire_use[par]['ba']['type'] == 'lin_mod':
                
                    self.Fire_vars[par]['ba'] = [x for x in self.Fire_use[par]['ba']['pars'].iloc[:,0].tolist() if x != 'Intercept']   
                
                elif self.Fire_use[par]['ba']['type'] == 'tree_mod':
                    
                    self.Fire_vars[par]['ba'] = [x for x in self.Fire_use[par]['ba']['pars'].iloc[:,1].tolist() if x != '<leaf>']
            
            

    
    ### Container for suppression pars
    def get_suppression_pars(self):
        
        pass
        
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
            self.Dist_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]


            ### combine numpy arrays to single pandas       
            self.Dist_dat  = pd.DataFrame.from_dict(dict(zip(self.Dist_vars, 
                           [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat])))
        
            ### do prediction
            self.Dist_vals = predict_from_tree_fast(dat = self.Dist_dat, 
                              tree = self.Dist_frame, struct = self.Dist_struct, 
                               prob = 'yprob.TRUE', skip_val = -3.3999999521443642e+38, na_return = 0)
                
        
            ### apply theta zero-ing out constraint
            self.Dist_vals = np.select([self.Dist_vals > self.model.p.theta], [self.Dist_vals], default = 0)
            
            
            ### bootstrapped version
        elif self.p.bootstrap == True:
            
            self.Dist_vals = []
            
            ### gather correct numpy arrays 4 predictor variables
            self.Dist_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]

            ### combine numpy arrays to single pandas       
            self.Dist_dat  = pd.DataFrame.from_dict(dict(zip(self.Dist_vars, 
                              [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat])))
        
            ### Parallel prediction
            boot_frame     = make_boot_frame(self)
            self.Dist_vals = parallel_predict(boot_frame, self.model.client, 'yprob.TRUE')
            self.Dist_vals = combine_bootstrap(self)
            

    
    #######################################################################
    
    ### Fire
    
    #######################################################################
    
    ################
    ### Fire use
    ################
    
    
    def fire_use(self):
        
       
    ####################################
                
    ### Prepare data
                
    ####################################
        
        self.Fire_dat = {}
        self.Fire_vals= {}

        probs_key     = {'bool': 'yprob.Presence', 
                         'ba'  : 'yval'} ### used for gathering final results
        
        
        ### gather numpy arrays 4 predictor variables
        for x in self.Fire_use.keys():
            
            for b in ['bool', 'ba']:  
          
              
                if b in self.Fire_vars[x].keys(): 
           
            
                    ### containers for fire outputs
                    self.Fire_dat[x]       = {} if not x in self.Fire_dat.keys() else self.Fire_dat[x]  
                    self.Fire_dat[x][b]    = []
                    self.Fire_vals[x]      = {} if not x in self.Fire_vals.keys() else self.Fire_vals[x]  
                    self.Fire_vals[x][b]   = []
            
                    temp_key = self.Fire_vars[x][b]
    
    
                    ### Gather relevant map data
                    for y in range(len(temp_key)):
            
                        temp_val = self.model.p.Maps[temp_key[y]][self.model.timestep, :, :] if len(self.model.p.Maps[temp_key[y]].shape) == 3 else self.model.p.Maps[temp_key[y]]
            
                        self.Fire_dat[x][b].append(temp_val)

                    ### combine predictor numpy arrays to a single pandas       
                    self.Fire_dat[x][b]  = pd.DataFrame.from_dict(dict(zip(self.Fire_vars[x][b], 
                           [z.reshape(self.model.p.xlen*self.model.p.ylen).data for z in self.Fire_dat[x][b]])))
        
        
    ####################################
                
    ### Make predictions
                
    ####################################
                
                ##########
                ### Tree
                ##########
                
                    if self.Fire_use[x][b]['type'] == 'tree_mod':
      
        
                        Fire_struct = define_tree_links(self.Fire_use[x][b]['pars'])

                        self.Fire_vals[x][b] = predict_from_tree_fast(dat =  self.Fire_dat[x][b], 
                              tree = self.Fire_use[x][b]['pars'], struct = Fire_struct, 
                               prob = probs_key[b], skip_val = -3.3999999521443642e+38, na_return = 0)
    
                ################
                ### Regression
                ################
                
                    elif self.Fire_use[x][b]['type'] == 'lin_mod':
    
                        self.Fire_vals[x][b] = deepcopy(self.Fire_dat[x][b])
                    
                        ### Mulitply data by regression coefs
                        for coef in self.Fire_vars[x][b]:
                        
                            self.Fire_vals[x][b][coef] = self.Fire_vals[x][b][coef] * self.Fire_use[x][b]['pars']['coef'].iloc[np.where(self.Fire_use[x][b]['pars']['var'] == coef)[0][0]]
                    
                        ### Add intercept
                        self.Fire_vals[x][b] = self.Fire_vals[x][b].sum(axis = 1) + self.Fire_use[x][b]['pars']['coef'].iloc[np.where(self.Fire_use[x][b]['pars']['var'] == 'Intercept')[0][0]]
                    
                        ### Link function
                        self.Fire_vals[x][b] = regression_transformation(regression_link(self.Fire_vals[x][b], 
                                                  link = self.Fire_use[x][b]['pars']['link'][0]), 
                                                                     transformation = self.Fire_use[x][b]['pars']['transformation'][0])
                        ### control for negative values
                        self.Fire_vals[x][b] = pd.Series([x if x > 0 else 0 for x in self.Fire_vals[x][b]])
                        
                #################################
                ### specified
                #################################
                    
                elif 'constant' in self.Fire_use[x][b].keys():              
                  
                    self.Fire_vals[x]      = {} if not x in self.Fire_vals.keys() else self.Fire_vals[x]  
                    self.Fire_vals[x][b]   = []
                    self.Fire_vals[x][b] = pd.Series([self.Fire_use[x][b]['constant']] * (self.model.p.ylen * self.model.p.xlen))
                    
                else:
                        
                    pass
              
            ### calculate burned area through bool & ba%
            self.Fire_vals[x] = self.Fire_vals[x]['bool'] * self.Fire_vals[x]['ba']
            
        ### Adjust for AFT specific constraints
        self.fire_constraints()
    
    
    def fire_constraints(self):
        
        ''' container for agent-specific fire constraints'''
        
        pass
    
    #######################
    
    ### Fire suppression
    
    #######################   
               
    def fire_suppression(self):
        
        ''' container for suppression actions'''
        
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
        
        
        
        