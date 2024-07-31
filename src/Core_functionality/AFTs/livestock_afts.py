# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""

import agentpy as ap
import pandas as pd
import numpy as np

from Core_functionality.AFTs.agent_class import AFT

###########################################################################################

### Livestock AFTs - Rangeland

###########################################################################################

class Pastoralist_r(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Rangeland'
        self.Habitat = {'Map': 'HDI', 
                        'Constraint': 0.7691141, 
                        'Constraint_type': 'lt'}  

        self.Fire_use = {'pasture': {'bool': 'tree_mod', 
                                 'ba': 'tree_mod', 
                                 'size': 6.55}}
        

    def fire_constraints(self):
        
        ### rangeland stocking constraint
        
        if self.model.p.Constraint_pars['Rangeland_stocking_contstraint'] == True:
        
            occupancy = [x.Dist_vals for x in self.model.agents if x.ls == self.ls] 
            occupancy = np.nansum(occupancy[0:1], axis = 0)
            
            ### should stocking rates be able to increase fire?
            if self.model.p.Constraint_pars['R_s_c_Positive'] == False:          
                
                occupancy = [x if x < 1 else 1.0 for x in occupancy.reshape(self.model.p.xlen*self.model.p.ylen)]
                
            self.Fire_vals['pasture'] = self.Fire_vals['pasture'] * occupancy
        
        else:
            
            pass


class Ext_LF_r(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Rangeland'
        self.Habitat = {'Map': 'HDI', 
                        'Constraint': 0.8886984, 
                        'Constraint_type': 'lt'}  
        
        self.Fire_use = {'pasture': {'bool': 'tree_mod', 
                                 'ba': 'tree_mod', 
                                 'size': 14.03571}}
        
                
    def fire_constraints(self):
        
        ### rangeland stocking constraint
        
        if self.model.p.Constraint_pars['Rangeland_stocking_contstraint'] == True:
        
            occupancy = [x.Dist_vals for x in self.model.agents if x.ls == self.ls] 
            occupancy = np.nansum(occupancy[0:1], axis = 0)
            
            ### should stocking rates be able to increase fire?
            if self.model.p.Constraint_pars['R_s_c_Positive'] == False:          
                
                occupancy = [x if x < 1 else 1.0 for x in occupancy.reshape(self.model.p.xlen*self.model.p.ylen)]
                
            self.Fire_vals['pasture'] = self.Fire_vals['pasture'] * occupancy
        
        else:
            
            pass
        
        
        
class Int_LF_r(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Rangeland'
        self.Habitat = {'Map': 'HDI', 
                        'Constraint': 0.59, 
                        'Constraint_type': 'gt'}   

        self.Fire_use = {'pasture': {'bool': 'tree_mod', 
                                 'ba': {'constant': 0.025}, 
                                 'size': 11.5}}
               
    def fire_constraints(self):
        
        ### rangeland stocking constraint
        
        if self.model.p.Constraint_pars['Rangeland_stocking_contstraint'] == True:
        
            occupancy = [x.Dist_vals for x in self.model.agents if x.ls == self.ls] 
            occupancy = np.nansum(occupancy[0:1], axis = 0)
            
            ### should stocking rates be able to increase fire?
            if self.model.p.Constraint_pars['R_s_c_Positive'] == False:          
                
                occupancy = [x if x < 1 else 1.0 for x in occupancy.reshape(self.model.p.xlen*self.model.p.ylen)]
                
            self.Fire_vals['pasture'] = self.Fire_vals['pasture'] * occupancy
        
        else:
            
            pass


class Abandoned_LF_r(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Post'
        self.ls  = 'Rangeland'
        self.Habitat = {'Map': 'TRI', 
                        'Constraint': 38.19444, 
                        'Constraint_type': 'gt'} 
        self.Fire_use = {}
        
        
###########################################################################################

### Livestock AFTs - Pasture

###########################################################################################

class Pastoralist_p(AFT):

    def setup(self):
        AFT.setup(self)
        self.afr = 'Pre'
        self.ls  = 'Pasture'
        self.Habitat = {'Map': 'HDI', 
                        'Constraint': 0.7691141, 
                        'Constraint_type': 'lt'}  

        self.Fire_use = {'pasture': {'bool': 'tree_mod', 
                                 'ba': 'tree_mod', 
                                 'size': 6.55}}


class Ext_LF_p(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Trans'
        self.ls  = 'Pasture'
              
        self.Fire_use = {'pasture': {'bool': 'tree_mod', 
                                 'ba': 'tree_mod', 
                                 'size': 14.03571}}
        
        self.Defor_size = 44.83549
        
        self.Nfer_use = {'tree': 'tree_mod'}
        
        
class Int_LF_p(AFT):
    
    def setup(self):
        AFT.setup(self)
        self.afr = 'Intense'
        self.ls  = 'Pasture'
        self.Habitat = 'None'
        
        self.Fire_use = {'pasture': {'bool': 'tree_mod', 
                                 'ba': {'constant': 0.025}, 
                                 'size': 11.5}}
        
        self.Defor_size = 65.35
        
        self.Nfer_use = {'tree': 'tree_mod'}

        
        
        
        