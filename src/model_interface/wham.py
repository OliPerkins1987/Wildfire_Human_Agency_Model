# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""


import agentpy as ap
import numpy as np
import pandas as pd


from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p
from Core_functionality.AFTs.forestry_afts  import Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts  import Hunter_gatherer, Recreationalist, SLM, Conservationist
from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

###################################################################

### Core model class

###################################################################


class WHAM(ap.Model):

    def setup(self):

        # Parameters
        self.xlen = self.p.xlen
        self.ylen = self.p.ylen

        # Create grid
        self.grid = ap.Grid(self, (self.xlen, self.ylen), track_empty=False)
        
        
        # Create land systems
        self.ls     = ap.AgentList(self, 
                       [y[0] for y in [ap.AgentList(self, 1, x) for x in self.p.LS]])
        
        # Create AFTs
        self.agents = ap.AgentList(self, 
                       [y[0] for y in [ap.AgentList(self, 1, x) for x in self.p.AFTs]])

    
    ##############################
    
    ### AFT distribution functions
    
    ###############################
    
    def allocate_X_axis(self):
        
        ### Gather X-axis vals from land systems       
        ls_scores    = dict(zip([type(x).__name__ for x in self.ls], 
                        [x.Dist_vals for x in self.ls]))
        
        #################################################
        ### Perform calculation to get X_axis
        #################################################
        
        ### Forestry
        ls_scores['Forestry'] =  ls_scores['Forestry'] * (1 - ls_scores['Nonex']['Forest']) * (1 - ls_scores['Unoccupied'])
        
        ### Non-ex & Unoccupied
        Open_vegetation                =  self.p.Maps['Mask'] - ls_scores['Cropland'] - ls_scores['Pasture'] - ls_scores['Rangeland'] - ls_scores['Forestry'] - ls_scores['Urban']
        Open_vegetation                =  np.array([x if x >=0 else 0 for x in Open_vegetation])
        ls_scores['Nonex']['Combined'] =  Open_vegetation * (ls_scores['Nonex']['Other'] / (ls_scores['Nonex']['Other'] + ls_scores['Unoccupied']))
        ls_scores['Unoccupied']        =  Open_vegetation * (ls_scores['Unoccupied'] / (ls_scores['Nonex']['Other'] + ls_scores['Unoccupied']))
        ls_scores['Nonex']             =  ls_scores['Nonex']['Combined']
        
        ### There is an issue with alignment of data sets giving LC > land mask (see forestry)
        ### Current workaround...
        
        ls_frame                       = pd.DataFrame(ls_scores)
        ls_frame['tot']                = self.p.Maps['Mask'] / ls_frame.sum(axis = 1) 
        ls_frame.iloc[:, 0:-1  ]       = ls_frame.iloc[:,0:-1].multiply(ls_frame.tot, axis="index")                            
        ls_frame                       = ls_frame.iloc[:, 0:-1].to_dict('series')
        
        ### reshape and stash
        self.X_axis                    =  dict(zip([x for x in ls_frame.keys()], 
                                            [np.array(x).reshape(self.ylen, self.xlen) for x in ls_frame.values()]))
        
    
    def allocate_Y_axis(self):
        
        ### Gather Y-axis scores from AFTs
        
        land_systems = [y for y in pd.Series([x for x in self.agents.ls]).unique()]
        afr_scores   = {}
    
    
        if type(land_systems) == str:
            land_systems = [land_systems] #catch the case where only 1 ls type
            
        for l in land_systems:
            
            ### get predictions
            afr_scores[l] = [x.Dist_vals for x in self.agents if x.ls == l]
                
            ### remove dupes - this only works with more than 1 AFR per LS
            unique_arr    = [np.array(x) for x in set(map(tuple, afr_scores[l]))]
            
            ### calculate total by land system by cell
            tot_y         = np.add.reduce(unique_arr)
            
            ### divide by total & reshape to world map
            afr_scores[l] = [np.array(x / tot_y).reshape(self.ylen, self.xlen) for x in unique_arr]
               
        
            ### Here - multiply Yscore by X-axis
            afr_scores[l] = dict(zip(list(dict.fromkeys([y for y in pd.Series([x.afr for x in self.agents if x.ls == l]).unique()])), 
                             [x * self.X_axis[l] for x in afr_scores[l]]))
        
        ### stash afr scores
        self.LFS = afr_scores
        
        
    def allocate_AFT(self):
        
        AFT_scores   = {}
        
        ### Loop through agents and assign fractional coverage
        for a in self.agents:
            
            if a.sub_AFT['exists'] == False:
            
                AFT_scores[type(a).__name__] = self.LFS[a.ls][a.afr]
                
            elif a.sub_AFT['exists'] == True:
                
                ### Where AFT is a fraction of a single LFS
                if a.sub_AFT['kind'] == 'Fraction':
                    
                    a.AFT_vals                   = np.array(a.AFT_vals).reshape(self.ylen, self.xlen)
                    AFT_scores[type(a).__name__] = self.LFS[a.ls][a.afr] * a.AFT_vals
                    
                    
                ### Where AFT is a whole LFS plus a fraction of another
                elif a.sub_AFT['kind'] == 'Addition':
                    
                    a.AFT_vals                   = np.array(a.AFT_vals).reshape(self.ylen, self.xlen)
                    AFT_scores[type(a).__name__] = self.LFS[a.ls][a.afr] + (self.LFS[a.sub_AFT['ls']][a.sub_AFT['afr']] * a.AFT_vals)
                
                
                ### Where AFT is a fraction of several LFS
                elif a.sub_AFT['kind'] == 'Multiple':
                    
                    AFT_scores[type(a).__name__] = np.zeros([self.ylen, self.xlen])
                
                    for i in range(len(a.sub_AFT['afr'])):
                        
                        temp_vals                    = np.array(a.AFT_vals[i]).reshape(self.ylen, self.xlen)
                        AFT_scores[type(a).__name__] = AFT_scores[type(a).__name__] + (self.LFS[a.sub_AFT['ls'][i]][a.sub_AFT['afr'][i]] * temp_vals)
                
        self.AFT_scores = AFT_scores
    
    #####################################################################################
    
    ### scheduler, recorder, end conditions

    #####################################################################################

    def update(self):
        
        pass
    
    
    def step(self):
        
        ### AFT distribution
        self.agents.compete()
        self.allocate_X_axis()
        self.allocate_Y_axis()
        
        ### Fire
        
    def end(self):
        
        pass
    
    
    

