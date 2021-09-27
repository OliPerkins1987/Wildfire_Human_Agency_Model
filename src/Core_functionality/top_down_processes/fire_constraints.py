# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:04:17 2021

@author: Oli
"""

import agentpy as ap
import numpy as np


class fuel_ct(ap.Agent):
    
    ''' constraint on fire due to limited fuel'''
    
    def constrain(self):
        
        ### Calculate soil constraint
        Soil = self.model.p.Maps['Baresoil'].data
        Soil = 1 - (Soil * (Soil>= self.model.p.Constraint_pars['Soil_threshold'])) #defaults mean bare soil cover

        
        ### multiple Soil constraint by relevant fire types
        self.model.Managed_fire['Pasture']     = self.model.Managed_fire['Pasture'] * Soil
        self.model.Managed_fire['Vegetation']  = self.model.Managed_fire['Vegetation'] * Soil


    def constrain_arson(self):
        
        ### Calculate soil constraint
        Soil = self.model.p.Maps['Baresoil'].data
        Soil = 1 - (Soil * (Soil>= self.model.p.Constraint_pars['Soil_threshold']))
        Soil = Soil.reshape(self.model.ylen * self.model.xlen)
        
        ### multiply arson by soil constraint
        self.model.Observers['arson'][0].Fire_vals = self.model.Observers['arson'][0].Fire_vals*Soil
        


class dominant_afr_ct(ap.Agent):
    
    ''' impact of dominant afr being industrial'''
    

    def constrain(self):
        
        ### calculate distribution of afr
        
        afr_res = {}
    
        for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
            afr_vals = []
    
            for ls in ['Nonex']:
        
                if afr in self.model.LFS[ls].keys():
                
                    afr_vals.append(self.model.LFS[ls][afr])
               
            afr_res[afr] = np.nansum(afr_vals, axis = 0)
            
            ### divide by nonex fraction
            afr_res[afr] = afr_res[afr] / self.model.X_axis['Nonex']
        
        ### This is interesting but needs more work
        ### Post-industrial areas boost pyrome mgmt & traditional fire use
        #Post    = np.nanargmax([x for x in afr_res.values()], axis = 0) == 3
        #Post    = Post * self.model.Managed_fire['Vegetation']
        #self.model.Managed_fire['Vegetation'] = self.model.Managed_fire['Vegetation'] + Post
        
        ### Zero intensive cases below dominance threshold
        afr_res['Intense'] = np.select([afr_res['Intense'] >= self.model.p.Constraint_pars['Dominant_afr_threshold']], 
                            [afr_res['Intense']], default = 0)
        
        ### calculate impact of dominant exclusionary afr
        Intense = np.nanargmax([x for x in afr_res.values()], axis = 0)
        Intense = ((Intense==2) * (1 - afr_res['Intense'])) + (Intense!=2 * 1)
                
        self.model.Managed_fire = dict(zip([x for x in self.model.Managed_fire.keys()], 
                                         [y*Intense for y in self.model.Managed_fire.values()]))

        
        
        
    def constrain_arson(self):

        afr_res = {}
    
        for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
            afr_vals = []
    
            for ls in ['Cropland', 'Rangeland', 'Pasture', 'Forestry', 'Nonex']:
        
                if afr in self.model.LFS[ls].keys():
                
                    afr_vals.append(self.model.LFS[ls][afr])
               
            afr_res[afr] = np.nansum(afr_vals, axis = 0)
            
        
        ### Zero intensive cases below dominance threshold
        afr_res['Intense'] = np.select([afr_res['Intense'] >= self.model.p.Constraint_pars['Dominant_afr_threshold']], 
                            [afr_res['Intense']], default = 0)
        
        ### calculate impact of dominant exclusionary afr
        Intense = np.nanargmax([x for x in afr_res.values()], axis = 0)
        Intense = ((Intense==2) * (1 - afr_res['Intense'])) + (Intense!=2 * 1)
        Intense = Intense.reshape(self.model.p.ylen * self.model.p.xlen)
                
        self.model.Observers['arson'][0].Fire_vals = self.model.Observers['arson'][0].Fire_vals*Intense

        ### Impact of Unoccupied regions 
        Unoc = np.array(1 - self.model.X_axis['Unoccupied']).reshape(self.model.p.ylen*self.model.p.xlen)
        self.model.Observers['arson'][0].Fire_vals = self.model.Observers['arson'][0].Fire_vals * Unoc


    
