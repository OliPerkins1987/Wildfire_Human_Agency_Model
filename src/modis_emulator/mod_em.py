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
            shape=(self.abm.p.ylen, self.abm.p.xlen))  # emulated BA

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
                    self.Managed_fire[i][a] = self.Managed_fire[i][a] * \
                        self.abm.AFT_scores[type(a).__name__]
                    self.Managed_fire[i][a] = np.array(self.Managed_fire[i][a]).reshape(
                        self.abm.p.ylen * self.abm.p.xlen)
                    self.Managed_igs[i][a] = self.Managed_fire[i][a] / \
                        (a.Fire_use[i]['size'] / 100)
                    self.Managed_big[i][a] = a.Fire_use[i]['size'] >= 21

        # !!! add INFERNO outputs

    def calc_big_fires(self):
        ''' add all fires bigger than 21ha as a fraction of the grid cell '''

        # ABM

        for i in self.Managed_fire.keys():

            for a in self.abm.agents:

                if i in a.Fire_vals.keys():

                    if self.Managed_big[i][a] == True:

                        self.map = np.nansum(
                            [self.map, self.Managed_fire[i][a]], axis=0)

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
        self.LC_pix['Vegetation'] = self.LC_pix['Mask'] - \
            self.LC_pix['Pasture'] - self.LC_pix['Arable']

        # Assign number of MODIS pixels to land covers
        for x in ['Arable', 'Pasture', 'Vegetation']:

            self.LC_pix[x] = self.LC_pix[x] * self.LC_pix['n_MODIS']
            self.LC_pix[x] = self.LC_pix[x].reshape(
                self.abm.p.ylen * self.abm.p.xlen)
            self.LC_pix[x] = np.array(
                [round(y) if y > 0 else 0 for y in np.array(self.LC_pix[x])])

    def null_cell(self, cell_numb):

        # is there any land in the cell?

        return(self.LC_pix['Mask'][cell_numb] > 0)

    def get_rate(self, fire_grid, Mod_grid):
        ''' calculates lambda for rate of fire per theoretical MODIS sub-cells '''

        return(fire_grid / Mod_grid)

    def poisson_fires(self, rate, n_cell, shuffle=True):
        ''' returns number of fires per cell following a poisson distribution'''

        pois_obj = sp.stats.poisson(rate)
        x_in     = [x for x in np.linspace(0, 1, n_cell + 2)] ### 0 and 1 will be -inf and inf!
        y_out    = [y for y in pois_obj.ppf(x_in[1:-1])]     ### finite probability space!

        if shuffle == True:

            random.shuffle(y_out)

        return(y_out)

    def assign_fires(self, cell_numb):
        ''' assigns numbers of fires to theoretical MODIS sub-cells by land cover type '''

        # arrays of length n_MODIS by land cover type

        cell = {}

        for lc in ['Arable', 'Pasture', 'Vegetation']:

            cell[lc] = np.zeros(self.LC_pix[lc][cell_numb])

            # loop through fire types then compile BA fractions for subcells

            for i in self.Managed_fire.keys():

                if self.abm.p.Fire_types[i] == lc:

                    for a in self.abm.agents:

                        if i in a.Fire_vals.keys() and self.LC_pix[lc][cell_numb] > 0:

                            fire_rate = self.get_rate(self.Managed_igs[i][a][cell_numb],
                                                      self.LC_pix[lc][cell_numb])

                            fire_dist = self.poisson_fires(
                                fire_rate, self.LC_pix[lc][cell_numb])

                            fire_dist = [x * a.Fire_use[i]['size']
                                         for x in fire_dist]

                            cell[lc]  = cell[lc] + np.array(fire_dist)


#####
# experiment
#####
em = modis_em(mod)
