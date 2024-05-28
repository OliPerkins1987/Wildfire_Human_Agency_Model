
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np
from copy import deepcopy

from Core_functionality.setup_tools.setup_funs import get_LU_pars
from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast, predict_from_tree_numpy
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation
from Core_functionality.Trees.parallel_predict import make_boot_frame, parallel_predict, combine_bootstrap
from Core_functionality.prediction_tools.predict_funs import get_LU_dat, predict_LU_behaviour


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
        self.Habitat = 'None'
        
        self.afr= ''        
        self.Fire_use = {}
        self.Fire_vars = {}
        
        self.Nfer_use  = {}
        self.Nfer_vars = {}
        
        
        
    ### gets AFT distribution parameters
    def get_dist_pars(self, AFT_dict):
        
        file_key         = str(self.ls + '/' + type(self).__name__)
        
        ### Distribution tree for LFS
        self.Dist_frame  = AFT_dict['AFT_dist'][file_key]
        self.Dist_struct = define_tree_links(self.Dist_frame)
        self.Dist_vars   = [x for x in self.Dist_frame.iloc[:,1].tolist() if x != '<leaf>']
        
        ### get numeric distributions of split thresholds and leaf probs
        if self.p.bootstrap == True:
            
            self.boot_Dist_pars   = {}
            self.boot_AFT_pars    = {}
                
            self.boot_Dist_pars['Thresholds']  = AFT_dict['Dist_pars']['Thresholds'][file_key]
            self.boot_Dist_pars['Probs']       = AFT_dict['Dist_pars']['Probs'][file_key]
        
            ### select number of boot parameter sets to use
            if self.p.numb_bootstrap != 'max':
            
                for i in range(len(self.boot_Dist_pars['Thresholds'])):
                
                    self.boot_Dist_pars['Thresholds'][i] = self.boot_Dist_pars['Thresholds'][i].iloc[0:self.model.p.numb_bootstrap, :]
            
                for i in range(len(self.boot_Dist_pars['Probs'])):
                
                    self.boot_Dist_pars['Probs'][i] = self.boot_Dist_pars['Probs'][i].iloc[0:self.model.p.numb_bootstrap, :]
                
        
    ### get Fire use parameters
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
                self.Fire_vars[par]['bool'] = get_LU_pars(self.Fire_use[par]['bool'])
            
            elif 'constant' in self.Fire_use[par]['bool'].keys():
                
                self.Fire_vars[par]         = {}
                self.Fire_vars[par]['bool'] = 'constant'
            
            ### get parameters for fire use degree (target % burned area)
        
            if self.Fire_use[par]['ba'] in ['lin_mod', 'tree_mod']: 
            
                self.Fire_use[par]['ba']   = {'type': self.Fire_use[par]['ba'], 
                                              'pars': self.p.AFT_pars['Fire_use']['ba'][par + '/' + type(self).__name__]}
                
                ###########################################
                ### extract parameter names
                ###########################################
                self.Fire_vars[par]['ba'] = get_LU_pars(self.Fire_use[par]['ba'])
                
            elif 'constant' in self.Fire_use[par]['ba'].keys():
                
                self.Fire_vars[par]['ba'] = 'constant'
     
        
    ### get parameters for nitrogen fertiliser use
    def get_Nfer_pars(self):
        
        for key, value in self.Nfer_use.items(): 
            
            if value in ['lin_mod', 'tree_mod']:
            
                self.Nfer_use[key] = {'type': self.Nfer_use[key], 
                                      'pars': self.p.AFT_pars['Nfer_use'][key][type(self).__name__]}
            
                self.Nfer_vars[key]= {}
                
                ###########################################
                ### extract parameter names
                ###########################################
                self.Nfer_vars[key] = get_LU_pars(self.Nfer_use[key])
        

    ### Container for suppression pars
    def get_suppression_pars(self):
        
        pass
        
    #########################################################################

    ### AFT Distribution

    #########################################################################    
    
    def get_habitat(self):
    
        ''' define habitat space of AFT'''    
    
        if self.Habitat != 'None':
            
            if len(self.model.p.Maps[self.Habitat['Map']].shape) == 2:
                
                tmp = self.model.p.Maps[self.Habitat['Map']].data
                
            elif len(self.model.p.Maps[self.Habitat['Map']].shape) == 3:
            
                tmp = self.model.p.Maps[self.Habitat['Map']][self.model.timestep, :, :].data
        
            if self.Habitat['Constraint_type'] == 'lt':
                
                tmp = (tmp <= self.Habitat['Constraint']).reshape(self.model.xlen * self.model.ylen)
                
            elif self.Habitat['Constraint_type'] == 'gt':
                
                tmp = (tmp >= self.Habitat['Constraint']).reshape(self.model.xlen * self.model.ylen)
            
            ### stash values
            self.Habitat_vals = tmp
    
    
    
    def compete(self):
        
        self.get_habitat()
        
        ''' 
        Competition between AFTs
        
        '''
        
            ### single set of parameter values
        if self.p.bootstrap != True:
        
            ### gather correct numpy arrays 4 predictor variables
            self.Dist_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]


            ### combine numpy arrays to single pandas       
            self.Dist_dat  = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat]).transpose()
        
            ### do prediction
            self.Dist_vals = predict_from_tree_numpy(dat = self.Dist_dat, 
                              tree = self.Dist_frame, split_vars = self.Dist_vars, struct = self.Dist_struct,
                               prob = 'yprob.TRUE', skip_val = -1e+10, na_return = 0)
                
        
            ### apply theta zero-ing out constraint
            self.Dist_vals = np.select([self.Dist_vals > self.model.p.theta], [self.Dist_vals], default = 0)
            
            ### apply habitat
            if self.Habitat != 'None':
                
                self.Dist_vals = self.Dist_vals * self.Habitat_vals
            
            
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
    
    #######################
    ### Fire use
    ####################### 
    
    def fire_use(self):
    
        fire_dat = {}
        self.Fire_vals= {}

        probs_key     = {'bool': 'yprob.Presence', 
                         'ba'  : 'yval'} ### used for gathering final results
        
        
        ### gather numpy arrays 4 predictor variables
        for x in self.Fire_use.keys():
            
            fire_dat[x]        = get_LU_dat(self, probs_key, self.Fire_vars[x])
            self.Fire_vals[x]  = predict_LU_behaviour(self, probs_key, 
                                 self.Fire_vars[x], fire_dat[x], 
                                 self.Fire_use[x])
            
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


 
    ##############################################################
    
    ### Nitrogen
    
    ##############################################################
    
    def nfer_use(self):
        
        if self.Nfer_use != {}:
        
            probs_key     = {'tree': 'yval', 
                         'lm'  : 'yval'}   
          
            nfer_dat       = get_LU_dat(self, probs_key, self.Nfer_vars)
        
            self.Nfer_vals = predict_LU_behaviour(aft = self, probs_dict = probs_key, 
                          vars_dict = self.Nfer_vars, dat = nfer_dat,
                          pars = self.Nfer_use)
        
            ### combine
            self.Nfer_vals = pd.DataFrame(self.Nfer_vals).mean(axis = 1)



##################################################################

### dummy agents for testing

##################################################################

class dummy_agent(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Test'
        self.ls  = 'Test'


        
        
        
        