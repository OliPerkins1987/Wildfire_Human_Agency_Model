# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:18:13 2022

@author: Oli
"""

import pandas as pd
import numpy as np
import agentpy as ap
import scipy as sp
import random


class modis_em():

    def __init__(self, abm):

        self.abm = abm
        self.map = np.zeros(
            shape=(self.abm.p.ylen * self.abm.p.xlen))  # emulated BA
        
        ### Treat shifting cultivation as arable (by land system)... 
        ### rather than vegetation (what gets burned)
        self.abm.p.Fire_types['cfp'] = 'Arable'
        
        # number of MODIS pixels per WHAM/INFERNO pixel
        self.n_MODIS = np.array(self.abm.Area / (0.5*0.5))

        # Land cover types dictionary
        self.LC_pix = {'n_MODIS': self.n_MODIS.reshape(self.abm.p.ylen * self.abm.p.xlen),
                       'Mask': self.abm.p.Maps['Mask'],
                       'Arable': np.zeros(shape=(self.abm.p.ylen * self.abm.p.xlen)),
                       'Pasture': np.zeros(shape=(self.abm.p.ylen * self.abm.p.xlen)),
                       'Vegetation': np.zeros(shape=(self.abm.p.ylen * self.abm.p.xlen))}


    def get_fire_vals(self):
        ''' get burned area frac by fire taxon (human, lightning, etc) '''

        long_area = self.abm.Area.reshape(self.abm.p.ylen*self.abm.p.xlen)

        self.Managed_fire = {}
        self.Managed_igs = {}
        self.Managed_big = {}

        for i in self.abm.p.Fire_types.keys():

            self.Managed_fire[i] = {}
            self.Managed_igs[i] = {}
            self.Managed_big[i] = {}

            # gather managed fire from abm

            for a in self.abm.agents:

                if i in a.Fire_vals.keys():

                    self.Managed_fire[i][a] = np.array(a.Fire_vals[i]).reshape(
                        self.abm.p.ylen, self.abm.p.xlen)
                    self.Managed_fire[i][a] = (self.Managed_fire[i][a] *
                        self.abm.AFT_scores[type(a).__name__])
                    self.Managed_fire[i][a] = np.array(self.Managed_fire[i][a]).reshape(
                        self.abm.p.ylen * self.abm.p.xlen)
                    self.Managed_igs[i][a] = (self.Managed_fire[i][a] /
                        (a.Fire_use[i]['size'] / 100)) * long_area
                    self.Managed_big[i][a] = a.Fire_use[i]['size'] >= 21

        # !!! add INFERNO outputs


    def divide_cells(self):
        ''' apportion numbers of MODIS pixels to different land use types '''

        X_axis = self.abm.X_axis

        # Assign fractions to land covers
        self.LC_pix['Arable'] = X_axis['Cropland'].reshape(
            self.abm.p.ylen * self.abm.p.xlen)
        self.LC_pix['Pasture'] = np.nansum(
            [X_axis['Pasture'], X_axis['Rangeland']], axis=0)
        self.LC_pix['Pasture'] = self.LC_pix['Pasture'].reshape(
            self.abm.p.ylen * self.abm.p.xlen)
        self.LC_pix['Vegetation'] = (self.LC_pix['Mask'] -
            self.LC_pix['Pasture'] - self.LC_pix['Arable'])

        # Assign number of MODIS pixels to land covers
        for x in ['Arable', 'Pasture', 'Vegetation']:

            self.LC_pix[x] = self.LC_pix[x] * self.LC_pix['n_MODIS']

            self.LC_pix[x] = np.array(
                [round(y) if y > 0 else 0 for y in np.array(self.LC_pix[x])])


    def not_null_cell(self, cell_numb):

        ''' is there any land in the cell? '''

        return(self.LC_pix['Mask'][cell_numb] > 0)


    def get_AFT_area(self, AFT, cell_numb):
    
        ''' find fractional coverage of AFT by cell'''    
    
        AFT_area = self.abm.AFT_scores[type(AFT).__name__].reshape(
            self.abm.p.ylen * self.abm.p.xlen)
        
        AFT_area = AFT_area[cell_numb]
        
        return(AFT_area)


    def get_rate(self, fire_grid, n_mod, AFT_area):
        
        ''' calculates lambda for rate of fire per theoretical MODIS sub-cells '''
        ''' calculation factors in proportion of a cell covered by the relevant lc '''        

                
        if AFT_area > 0:
        
        ### numb fires / n MODIS pixels (lc) / nMODIS total weighted by AFT_coverage    
        
            return(fire_grid / (n_mod * AFT_area))
        
        else:
            
            return 0.0


    def poisson_fires(self, rate, n_cell, AFT_area, cell_numb, shuffle=True):
        
        ''' returns number of fires per cell following a poisson distribution'''
        ''' Logic is that n_fires = pois(rate) if AFT > 0, else 0 '''

        ### make poisson object
        pois_obj = sp.stats.poisson(rate)
        
        ### define fraction of MODIS pixels for given AFT
        lc_frac   = n_cell / self.LC_pix['n_MODIS'][cell_numb]
        
        ### is AFT present?
        
        if (n_cell * (AFT_area/lc_frac)) > 0:
            
            ### yes - calc AFT area and proceed
            AFT_cells = round(n_cell * (AFT_area/lc_frac))
        
        else:
            
            ### no - return zeros
            return([x for x in np.zeros(n_cell)])
        
        ### sample quantiles of poisson
        x_in     = [x for x in np.linspace(0, 1, AFT_cells + 2)] ### 0 and 1 will be -inf and inf!
        y_out    = [y for y in pois_obj.ppf(x_in[1:-1])]         ### finite probability space!
        
        ### append 0s for AFT not present within the land cover type
        
        for x in range((n_cell - AFT_cells)):
            
            y_out.append(0.0) 
        
        if shuffle == True:

            random.shuffle(y_out)

        return(y_out)


    def assign_fires(self, cell_numb):
        ''' assigns numbers of fires to theoretical MODIS sub-cells by land cover type '''

        # arrays of length n_MODIS by land cover type

        cell = {}
        cn   = int(cell_numb)

        for lc in ['Arable', 'Pasture', 'Vegetation']:

            cell[lc] = np.zeros(int(self.LC_pix[lc][cn]))

            # loop through fire types then compile BA fractions for subcells

            for i in self.Managed_fire.keys():
                
                ### only fire types for relevant land cover
                
                if self.abm.p.Fire_types[i] == lc:                    

                    for a in self.abm.agents:
                        
                        ### only cells with some of the relevant lc
                        
                        if i in a.Fire_vals.keys() and self.LC_pix[lc][cn] > 0:

                            ### fractional coverage of AFT
                            a_area    = self.get_AFT_area(a, cn)
                            
                            ### rate of poisson distribution                            
                            fire_rate = self.get_rate(self.Managed_igs[i][a][cn],
                                                      self.LC_pix['n_MODIS'][cn], 
                                                      a_area)

                            ### Distribute fires using poisson process
                            fire_dist = self.poisson_fires(
                                fire_rate, self.LC_pix[lc][cn], a_area, cn)

                            ### burned area from n fires
                            fire_dist = [x * a.Fire_use[i]['size']
                                         for x in fire_dist]
                            
                            cell[lc]  = cell[lc] + np.array(fire_dist)
        
        return(cell)
        
    
    def distribute_ba(self, cell_dict):
        
        ### take cells with BA > 21ha (MODIS pixel) & reallocate
        ### assumes BA != >100% in a given month
        ### spillover fire, including larger fires, moves to contiguous cells
        
        cell = cell_dict
        
        for lc in ['Arable', 'Pasture', 'Vegetation']:
        
                
            ### only conduct smoothing if overall ba in lc is less than 100%!
            ### (unlikely edge case)
            
            if np.nansum(cell[lc]* cell[lc].shape[0]) >  (21 * cell[lc].shape[0]):
                
                pass
        
            else: 
                
                while any(cell[lc] > 21):
        
                    for i in range(cell[lc].shape[0]):
                        
                        ### cell has ba > 100%?
                        
                        if cell[lc][i] > 21:
                        
                            ### assign overflow to contiguous cell(s)
                        
                            diff     = cell[lc][i] - 21
                            new_cell = (i + 1) if (i + 1) <= cell[lc].shape[0] else 0
                            cell[lc][new_cell] = cell[lc][new_cell] + diff 
                            
                            ### set ba of original cell to 100%
                            
                            cell[lc][i] = 21.0
                
        return(cell)
    
    
    def filter_ba_MODIS(self, cell_dict):
        
        '''filter out partially burned cells which MODIS will not detect'''
        
        cell = cell_dict
        
        for lc in ['Arable', 'Pasture', 'Vegetation']:
            
            for i in range(cell[lc].shape[0]):
                
               cell[lc][i] = 0 if cell[lc][i] < 21 else cell[lc][i]
        
        return(cell)
    
    
    def emulate(self):
        
        cells = []
        
        for i in range(self.map.shape[0]):
    
            ### skip ocean !       
    
            if self.not_null_cell(i):
                
                ### assign fires to MODIS pixels
                temp_cell = self.assign_fires(i)
                
                ### prevent >100% BA in a month
                temp_cell = self.distribute_ba(temp_cell)
                
                ### apply MODIS filter
                temp_cell = self.filter_ba_MODIS(temp_cell)
                
                ### add totals of different land systems by WHAM cell
                temp_cell = dict(zip([x for x in temp_cell.keys()], 
                                 [np.nansum(y) for y in temp_cell.values()]))
                
                            
            else:
                
                ### Cells with no land
                temp_cell = {'Arable': 0.0, 'Pasture': 0.0, 'Vegetation':0.0}
        
            ### gather together
            cells.append(temp_cell)
            
            print(i)
            
        self.emulated_BA = cells


#####
# experiment
#####
em3 = modis_em(mod)
em3.get_fire_vals()
em3.divide_cells()
em3.emulate()


res = [x['Pasture'] + x['Arable'] + x['Vegetation'] for x in em3.emulated_BA]
res = (np.array(res) / 100) / em.abm.Area.reshape(144*192)
