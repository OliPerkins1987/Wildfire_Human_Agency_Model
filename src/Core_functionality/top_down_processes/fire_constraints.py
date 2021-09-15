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
        Soil = 1 - (Soil * (Soil> 0.5))

        
        ### multiple Soil constraint by relevant fire types
        self.model.Managed_fire['Pasture']     = self.model.Managed_fire['Pasture'] * Soil
        self.model.Managed_fire['Vegetation']  = self.model.Managed_fire['Vegetation'] * Soil


    def constrain_arson(self):
        
        ### Calculate soil constraint
        Soil = self.model.p.Maps['Baresoil'].data
        Soil = 1 - (Soil * (Soil> 0.5))
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
    
            for ls in ['Rangeland', 'Pasture' ,'Forestry', 'Nonex']:
        
                if afr in self.model.LFS[ls].keys():
                
                    afr_vals.append(self.model.LFS[ls][afr])
               
            afr_res[afr] = np.nansum(afr_vals, axis = 0)
        
        
        ### calculate impact of dominant exclusionary afr
        Intense = np.nanargmax([x for x in afr_res.values()], axis = 0)
        Intense = ((Intense==2) * (1 - afr_res['Intense'])) + (Intense!=2 * 1)
        
        self.model.Managed_fire = dict(zip([x for x in self.model.Managed_fire.keys()], 
                                         [y*Intense for y in self.model.Managed_fire.values()]))


    def constrain_arson(self):

        afr_res = {}
    
        for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
            afr_vals = []
    
            for ls in ['Rangeland', 'Pasture' ,'Forestry', 'Nonex']:
        
                if afr in self.model.LFS[ls].keys():
                
                    afr_vals.append(self.model.LFS[ls][afr])
               
            afr_res[afr] = np.nansum(afr_vals, axis = 0)
        
        
        ### calculate impact of dominant exclusionary afr
        Intense = np.nanargmax([x for x in afr_res.values()], axis = 0)
        Intense = ((Intense==2) * (1 - afr_res['Intense'])) + (Intense!=2 * 1)
        Intense = Intense.reshape(self.model.p.ylen * self.model.p.xlen)
        
        self.model.Observers['arson'][0].Fire_vals = self.model.Observers['arson'][0].Fire_vals*Intense


    
