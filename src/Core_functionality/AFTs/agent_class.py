
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np
from copy import deepcopy

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_numpy
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
        sub_AFT = does this AFT subdivide an LFS? Kind is one of fraction, addition
                    
        '''  
      
        self.ls = ''
        self.afr= ''
        self.sub_AFT = {'exists': False,
                        'kind'  : ''}
        
        self.Fire_use = {}
        self.Fire_vars = {}
        

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
        
            ### gather numpy arrays of predictor variables
            self.Dist_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]

            ### combine numpy arrays to single pandas       
            self.Dist_dat  = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat]).transpose()
        
            ### do prediction
            self.Dist_vals = predict_from_tree_numpy(dat = self.Dist_dat, 
                              tree = self.Dist_frame, split_vars = self.Dist_vars, struct = self.Dist_struct,
                               prob = 'yprob.TRUE', skip_val = -1e+10, na_return = 0)
                
        
            ### apply theta zero-ing out constraint
            self.Dist_vals = np.select([self.Dist_vals > self.model.p.theta], [self.Dist_vals], default = 0)
            
            
            ### bootstrapped version
        elif self.p.bootstrap == True:
            
            self.Dist_vals = []
            
            ### gather correct numpy arrays 4 predictor variables
            self.Dist_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.Dist_vars]

            ### combine numpy arrays to single pandas       
            self.Dist_dat  = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Dist_dat]).transpose()
        
            ### Parallel prediction
            boot_frame     = make_boot_frame(self)
            self.Dist_vals = parallel_predict(boot_frame, self.model.client, 'yprob.TRUE', self.Dist_vars)
            self.Dist_vals = combine_bootstrap(self)
            
        
    def sub_compete(self):
        
        ''' Competition between AFTs within each LFS '''
        
        ### 1 parameter version
        
        if self.sub_AFT['exists'] == True and self.p.bootstrap == False:
        
            if self.sub_AFT['kind'] != 'Multiple':    
        
                ### gather numpy arrays of predictor variables
                self.AFT_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.AFT_vars]

                ### combine numpy arrays to single pandas       
                self.AFT_dat  = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat]).transpose()
            
                ### do prediction
                self.AFT_vals = predict_from_tree_numpy(dat = self.AFT_dat, 
                                  tree = self.AFT_frame, split_vars = self.AFT_vars, struct = self.AFT_struct,
                                   prob = type(self).__name__, skip_val = -1e+10, na_return = 0)
 
    
            elif self.sub_AFT['kind'] == 'Multiple':
                
                self.AFT_dat  = []
                self.AFT_vals = []
                
                for i in range(len(self.sub_AFT['afr'])): 
            
                    ### gather correct numpy arrays 4 predictor variables
                    self.AFT_dat.append([self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.AFT_vars[i]])
        
                    ### combine numpy arrays to single pandas       
                    self.AFT_dat[i]   = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat[i]]).transpose()
            
                    ### do prediction - these are added together in the WHAM AFT allocate routine
                    self.AFT_vals.append(predict_from_tree_numpy(dat = self.AFT_dat[i], 
                                      tree = self.AFT_frame[i], split_vars = self.AFT_vars[i], struct = self.AFT_struct[i],
                                       prob = type(self).__name__, skip_val = -1e+10, na_return = 0))
       
        
        ### bootstrapped parameters
        
        elif self.sub_AFT['exists'] == True and self.p.bootstrap == True:
        
            if self.sub_AFT['kind'] != 'Multiple':    
        
                ### gather correct numpy arrays 4 predictor variables
                self.AFT_dat   = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.AFT_vars]
        
                ### combine numpy arrays to single pandas       
                self.AFT_dat   = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat]).transpose()
            
                ### Parallel prediction, no theta threshold for sub-splits           
                boot_frame      = make_boot_frame_AFT(self)
                av              = parallel_predict(boot_frame, self.model.client, type(self).__name__, self.AFT_vars)
                self.AFT_vals   = pd.DataFrame(np.column_stack(av)).mean(axis = 1).to_list()
                       
        
            elif self.sub_AFT['kind'] == 'Multiple':
        
                self.AFT_dat  = []
                self.AFT_vals = []
                
                for z in range(len(self.sub_AFT['afr'])): 
            
                    self.AFT_vals.append([])
                                    
                    ### gather correct numpy arrays 4 predictor variables
                    self.AFT_dat  = [self.model.p.Maps[x][self.model.timestep, :, :] if len(self.model.p.Maps[x].shape) == 3 else self.model.p.Maps[x] for x in self.AFT_vars[z]]
        
                    ### combine numpy arrays to single pandas       
                    self.AFT_dat  = np.array([x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.AFT_dat]).transpose()
                    
                    ### do parallel prediction
                    boot_frame      = make_boot_frame_AFT(self, par_set = z)
                    av              = parallel_predict(boot_frame, self.model.client, type(self).__name__, self.AFT_vars[z])
                    self.AFT_vals[z]= pd.DataFrame(np.column_stack(av)).mean(axis = 1).to_list()
                    
        else:
            
            self.AFT_vals  = 'None'
    
    
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
       

    ####################################
                
    ### Make predictions
                
    ####################################

                ##########
                ### Tree
                ##########
                
                    if self.Fire_use[x][b]['type'] == 'tree_mod':
        
                        ### combine predictor numpy arrays to a single pandas       
                        self.Fire_dat[x][b]  = np.array([x.reshape(
                            self.model.p.xlen*self.model.p.ylen).data for x in self.Fire_dat[x][b]]).transpose()                
        
                        Fire_struct = define_tree_links(self.Fire_use[x][b]['pars'])

                        self.Fire_vals[x][b] = predict_from_tree_numpy(dat =  self.Fire_dat[x][b], 
                          tree = self.Fire_use[x][b]['pars'], split_vars = self.Fire_vars[x][b], 
                          struct = Fire_struct, prob = probs_key[b], 
                          skip_val = -1e+10, na_return = 0)
                               
    
                ################
                ### Regression
                ################
                
                    elif self.Fire_use[x][b]['type'] == 'lin_mod':
                        
                        self.Fire_dat[x][b] = pd.DataFrame.from_dict(dict(zip(self.Fire_vars[x][b], 
                                          [x.reshape(self.model.p.xlen*self.model.p.ylen).data for x in self.Fire_dat[x][b]])))
                        
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
                        self.Fire_vals[x][b] = np.array([x if x > 0 else 0 for x in self.Fire_vals[x][b]])
                        
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
        
        
        
        