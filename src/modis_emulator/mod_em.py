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

### for evaluation
import os
import math

class modis_em():

    def __init__(self, abm, dgvm):

        random.seed(1987) #for poisson fire assignment
        self.abm = abm
        self.dgvm= dgvm
        
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
        
        # Flammability
        self.Flammability    = np.nansum(self.dgvm['Flammability'] * self.dgvm['PFT'], axis = 0)
    
    ###################################################################
    
    ### Setup functions
    
    ###################################################################
    
    def setup_emulate(self):
        
        '''run emulate setup'''
        
        ### set up emulation
        self.get_fire_vals()
        self.divide_cells()
        
        ### managed fire
        self.constrain_managed_fire()
        
        ### lightning & unmanaged fire
        self.calc_bare_soil()
        self.calc_lightning_ba_ls()
    
    
    def get_fire_vals(self):
        
        ''' get burned area frac by fire taxon (human, lightning, etc) '''

        long_area = self.abm.Area.reshape(self.abm.p.ylen*self.abm.p.xlen)
        
        ###########################################################
        ### ABM managed fire
        ###########################################################
        
        self.Managed_fire = {}
        self.Managed_igs  = {}

        for i in self.abm.p.Fire_types.keys():

            self.Managed_fire[i] = {}
            self.Managed_igs[i]  = {}
            
            # gather managed fire from abm

            for a in self.abm.agents:

                if i in a.Fire_vals.keys():

                    self.Managed_fire[i][a] = np.array(a.Fire_vals[i]).reshape(
                        self.abm.p.ylen, self.abm.p.xlen)
                    self.Managed_fire[i][a] = (self.Managed_fire[i][a] *
                        self.abm.AFT_scores[type(a).__name__])
                    self.Managed_fire[i][a] = np.array(self.Managed_fire[i][a]).reshape(
                        self.abm.p.ylen * self.abm.p.xlen)
                    self.Managed_igs[i][a]  = (self.Managed_fire[i][a] /
                        (a.Fire_use[i]['size'] / 100)) * long_area ##size in km2
        
        ################################################
        # Lightning fire
        ################################################
        
        Lightning_fire = np.nansum(self.abm.Area *self.dgvm['Lightning_fires'] * 
                                   self.dgvm['PFT_ba'] * self.dgvm['PFT'] * self.Flammability, 
                                        axis = 0)
        
        Lightning_igs  = self.dgvm['Lightning_fires'] * self.abm.Area
               
        self.Lightning = {'ba_frac': Lightning_fire * (1-self.abm.Suppression), 
                          'igs'    : Lightning_igs * (1 - self.abm.Suppression)}
        
        ################################################
        ### Unmanaged anthropogenic fire
        ################################################
        
        Flam_norm      = self.dgvm['Flammability'] / np.nanmean(np.select([self.dgvm['Flammability'] == 0, self.dgvm['Flammability'] >= 1], 
                                              [np.nan, 1], default = self.dgvm['Flammability']))
                
        UI             =  (self.abm.Arson + self.abm.Background_ignitions + self.abm.Escaped_fire) if self.abm.p['escaped_fire'] == True else (self.abm.Arson + self.abm.Background_ignitions)
        
        ### suppression
        Unmanaged_igs  =  UI * (1-self.abm.Suppression)
        
        self.Unmanaged_fire = np.array(self.abm.Area * Unmanaged_igs * np.nansum(Flam_norm, axis= 0) * np.nansum(
                               self.dgvm['PFT_ba'] * self.dgvm['PFT'], 
                                        axis = 0)).reshape(self.abm.p.xlen * self.abm.p.ylen)
        
    def constrain_managed_fire(self):
        
        ''' apply WHAM top-down fire constraints by land user '''
        
        for c in self.abm.Observers.values():
            
            if 'ct' in type(c[0]).__name__:
                
                constraint = c[0].Constraint.reshape(self.abm.p.ylen * self.abm.p.xlen)
                
                for k in self.Managed_fire.keys():
                
                    self.Managed_fire[k] = dict(zip([x for x in self.Managed_fire[k].keys()], 
                                         [y*constraint for y in self.Managed_fire[k].values()]))

                    self.Managed_igs[k] = dict(zip([x for x in self.Managed_igs[k].keys()], 
                                         [y*constraint for y in self.Managed_igs[k].values()]))

    
    def calc_lightning_ba_ls(self):
        
        ''' calculates BA per pft by WHAM land systems'''
        
        ### need to divide pasture into rangeland & managed pasture
        range_frac     = [x.Dist_vals for x in self.abm.ls if type(x).__name__ == 'Rangeland'][0]
        pasture_frac   = [x.Dist_vals for x in self.abm.ls if type(x).__name__ == 'Pasture'][0]
        
        range_frac     = range_frac / (range_frac + pasture_frac)
        pasture_frac   = pasture_frac / (range_frac + pasture_frac)
        
        ### calculate sizes
        size           = {}
        
        ### constant size for arable of 0.4 km2 (set to 0 currently)
        size['Arable'] = np.array([0 * 100] * (self.abm.ylen*self.abm.xlen)).reshape(self.abm.ylen, self.abm.xlen)
            
        ############################
        ### open vegetation
        ############################
        
        # ba per pft
        size['Vegetation'] = (self.dgvm['PFT_ba'][[0, 1, 2, 3, 4, 5, 8, 11, 12], :, :] * self.abm.Area) * self.dgvm['PFT'][[0, 1, 2, 3, 4, 5, 8, 11, 12], :, :]
        
        # divide by fraction of cell occupied by relevant PFTs
        size['Vegetation'] = np.nansum(size['Vegetation'], axis = 0) / np.nansum(self.dgvm['PFT'][[0, 1, 2, 3, 4, 5, 8, 11, 12], :, :], axis = 0)
        
        # convert to hectares
        size['Vegetation'] = size['Vegetation'] * 100
        
        ##################################################
        ### pasture - adjusted for rangeland fraction
        ##################################################
        
        size['Pasture']= 100 * np.array([3.2] * (self.abm.ylen*self.abm.xlen)).reshape(self.abm.ylen, self.abm.xlen)
        size['Pasture']= (size['Pasture'] * pasture_frac.reshape(self.abm.ylen, self.abm.xlen)) + (
                                size['Vegetation'] * range_frac.reshape(self.abm.ylen, self.abm.xlen))
        
        ###################################################
        ### Adjust sizes for bare soil
        ###################################################        
        
        for k in size.keys():
            
            size[k] = size[k] * (1-self.Bare_soil[k])
        
        
        self.Lightning['size'] = size
        
      
    def calc_bare_soil(self):
        
        ''' needed to convert JULES PFTs to WHAM land systems'''
        ''' allocates bare soil from PFT to each WHAM land system'''
        
        Arable_JULES = self.dgvm['PFT'][6, :, :] + self.dgvm['PFT'][9, :, :]
        Arable_Soil  = self.abm.X_axis['Cropland'] - Arable_JULES
        Arable_Soil  = np.select([Arable_Soil < 0], [0], default = Arable_Soil)
        
        Pasture_JULES = self.dgvm['PFT'][7, :, :] + self.dgvm['PFT'][10, :, :]
        Pasture_Soil  = self.abm.X_axis['Pasture'] - Pasture_JULES
        Pasture_Soil  = np.select([Pasture_Soil < 0], [0], default = Pasture_Soil)
        
        Vegetation_Soil = self.dgvm['Bare_soil'] - Arable_Soil - Pasture_Soil
        Vegetation_Soil = np.select([Vegetation_Soil < 0], [0], default = Vegetation_Soil)
        
        self.Bare_soil  = {'Arable': Arable_Soil, 
                           'Pasture': Pasture_Soil, 
                           'Vegetation': Vegetation_Soil}
        
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
    
    
    ##########################################################################
    
    ### Functions for running emulation
    
    ##########################################################################
    

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
        lc_frac  = n_cell / self.LC_pix['n_MODIS'][cell_numb]
        
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


    def assign_managed_fires(self, cell_numb):
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
        
    
    
    
    def assign_unmanaged_fires(self, cell_numb):
    
        ''' defines distribution of lightning fires'''    
    
    
        ### assign fire numbers by land system
        cell    = {}
        cn      = int(cell_numb)
        lc_frac = {'Arable' : (self.LC_pix['Arable'][cn] / self.LC_pix['n_MODIS'][cn]), 
                   'Pasture': self.LC_pix['Pasture'][cn] / self.LC_pix['n_MODIS'][cn], 
                   'Vegetation' : self.LC_pix['Vegetation'][cn] / self.LC_pix['n_MODIS'][cn]}
        
        ### combine lightning and unmanaged & assign to land covers
        unmanaged_fires = dict(zip([x for x in lc_frac.keys()], 
                              [y * (self.Unmanaged_fire[cn] + self.Lightning['igs'].reshape(self.abm.ylen * self.abm.xlen)[cn]) for y in lc_frac.values()]))
        
        ### round to integer
        unmanaged_fires = dict(zip([x for x in unmanaged_fires.keys()], 
                                   [round(y) if np.isnan(y) == False else 0 for y in unmanaged_fires.values()]))
        
       
        ### distribute in cell
        for lc in ['Arable', 'Pasture', 'Vegetation']:
            
            cell[lc] = self.poisson_fires(
                        unmanaged_fires[lc] / self.LC_pix[lc][cn], 
                            self.LC_pix[lc][cn], lc_frac[lc], cn)
        
            ### calc fire size
            cell[lc] = np.array([x * (self.Lightning['size'][lc].reshape(
                              self.abm.p.ylen*self.abm.p.xlen)[cn]) for x in cell[lc]])


        return(cell)

    
    def distribute_ba(self, cell_dict):
        
        ### take cells with BA > 21ha (MODIS pixel) & reallocate
        ### assumes BA != >100% in a given month
        ### spillover fire, including larger fires, moves to contiguous cells
        
        cell = cell_dict
        
        for lc in ['Arable', 'Pasture', 'Vegetation']:
        
                
            ### only conduct smoothing if overall ba in lc is less than 100%!
            
            if np.nansum(cell[lc]) >  (21 * cell[lc].shape[0]):
                
                pass
        
            else: 
                
                while any(cell[lc] > 21):
        
                    for i in range(cell[lc].shape[0]):
                        
                        ### cell has ba > 100%?
                        
                        if cell[lc][i] > 21:
                        
                            ### assign overflow to contiguous cell(s)
                        
                            diff     = cell[lc][i] - 21
                            new_cell = (i + 1) if (i + 1) < cell[lc].shape[0] else 0
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
        
        ''' run emulation'''
        
        cells = []
                
        ### run emulation
        for i in range(self.map.shape[0]):
    
            ### skip ocean!       
    
            if self.not_null_cell(i):
                
                ### assign managed fires to MODIS pixels
                temp_managed      = self.assign_managed_fires(i)
                
                ### assign unmanaged fires to MODIS pixels
                temp_um           = self.assign_unmanaged_fires(i) 
                
                ### combine
                temp_cell = dict(zip([x for x in temp_managed.keys()], 
                            [x + y for x, y in zip(temp_managed.values(), temp_um.values())]))
                
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
            
            self.cells = cells
            
            if i % 1000 == 0:
                
                print('Emulation ', round((i / 27648) * 100, 2), '% completed')

        total = [val.values() for val in cells]
        total = np.array([sum(x) for x in total]).reshape(self.abm.p.ylen, self.abm.p.xlen)

        ### reassemble array
        self.emulated_BA = {'Arable': np.array([x['Arable'] for x in cells]).reshape(
                                    self.abm.p.ylen, self.abm.p.xlen), 
                            'Pasture': np.array([x['Pasture'] for x in cells]).reshape(
                                    self.abm.p.ylen, self.abm.p.xlen), 
                            'Vegetation': np.array([x['Vegetation'] for x in cells]).reshape(
                                    self.abm.p.ylen, self.abm.p.xlen), 
                            'Total': total}
        
        ### convert hectares -> km2 -> BA fraction
        self.emulated_BA = dict(zip([x for x in self.emulated_BA.keys()], 
                            [y / 100 / self.abm.Area for y in self.emulated_BA.values()]))
        
        ### switch shifting cultivation back to vegetation
        self.abm.p.Fire_types['cfp'] = 'Vegetation'
        


    def em_eval(self, tc):
        
        '''utility for analysing outputs'''
        
        ### combine land systems
        combined_cell = np.concatenate([x for x in tc.values()], axis = 0)
        
        ### round to 100 for arranging
        combined_cell = combined_cell[0:math.floor(combined_cell.shape[0] / 100)*100]
        
        ### arrange the combined cell into a square
        t=np.arange(2,combined_cell.shape[0],1)
        t=t[combined_cell.shape[0]%t==0]

        middle = float(len([x for x in t]))/2
        if middle == 0:
            return(np.array([[0], [0]]))
        elif middle % 2 != 0:
            t = t[int(middle - .5)]
        else:
            t = t[int(middle-1)]
            
        combined_cell = combined_cell.reshape(int(t), 
                         int(combined_cell.shape[0]/t))
        
        return(combined_cell)
    
        
#######################################

### run emulator

#######################################

if __name__ == "__main__":

    ### instantiate
    em = modis_em(mod, INFERNO)

    ### setup
    em.setup_emulate()

    ### go
    em.emulate()


