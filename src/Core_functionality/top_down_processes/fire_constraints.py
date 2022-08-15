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
        
        pars = self.model.p.Constraint_pars['Soil_threshold']
        
        ### Calculate vegetation absence constraint
        Soil = self.model.p.Maps['NPP'].data[self.model.timestep, :, :]
        Soil = ((Soil - pars['min']) / (pars['max'] - pars['min']))
        
        Soil = Soil + (self.model.p.Maps['NPP'].data[self.model.timestep, :, :] > pars['median'])
        Soil = np.select([Soil < 0, Soil > 1], [0, 1], default=Soil)
        
        ### constrain all apart from crop fires
        f_con = [x for x in self.model.Managed_fire.keys() if x not in ['crb', 'Cropland', 'Arable']]
        self.model.Managed_fire = dict(zip([x for x in self.model.Managed_fire.keys()], 
                                   [self.model.Managed_fire[x] * 
                                    Soil if x in f_con else self.model.Managed_fire[x] for x in self.model.Managed_fire.keys()]))
        
        
        ### store for easy analysis & use with emulator
        self.Constraint = Soil

    def constrain_arson(self):
        
        pars = self.model.p.Constraint_pars['Soil_threshold']
        
        ### Calculate vegetation absence constraint
        Soil = self.model.p.Maps['NPP'].data[self.model.timestep, :, :]
        Soil = ((Soil - pars['min']) / (pars['max'] - pars['min']))
        
        Soil = Soil + (self.model.p.Maps['NPP'].data[self.model.timestep, :, :] > pars['median'])
        Soil = np.select([Soil < 0, Soil > 1], [0, 1], default=Soil)
        
        ### reshape
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
    
            for ls in ['Cropland', 'Pasture', 'Rangeland', 'Forestry','Nonex']:
        
                if afr in self.model.LFS[ls].keys():
                
                    afr_vals.append(self.model.LFS[ls][afr])
               
            afr_res[afr] = np.nansum(afr_vals, axis = 0)

        
        
        ### Zero intensive cases below dominance threshold
        afr_res['Intense'] = np.select([afr_res['Intense'] >= self.model.p.Constraint_pars['Dominant_afr_threshold']], 
                            [afr_res['Intense']], default = 0)
        
        ### calculate impact of dominant exclusionary afr
        Intense = np.nanargmax([x for x in afr_res.values()], axis = 0)
        Intense = ((Intense==2) * (1 - afr_res['Intense'])) + (Intense!=2 * 1)
                
        self.model.Managed_fire = dict(zip([x for x in self.model.Managed_fire.keys()], 
                                         [y*Intense for y in self.model.Managed_fire.values()]))

        ### store for easy analysis & use with emulator
        self.Constraint = Intense
        
        
    def constrain_arson(self):

        afr_res = {}
    
        for afr in ['Pre', 'Trans', 'Intense', 'Post']:
    
            afr_vals = []
    
            for ls in ['Cropland', 'Pasture', 'Rangeland', 'Forestry','Nonex']:
        
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
