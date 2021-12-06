# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:54:20 2021

@author: Oli
"""

import netCDF4 as nc
import numpy as np


def write_nc(fn, vals, mod, annual = True):

    ds = nc.Dataset(fn, 'w', format='NETCDF4')
    
    if annual == True:
    
        time = ds.createDimension('time', 1)
    
    else:
        
        time = ds.createDimension('time', (mod.p.end_run - mod.p.start_run))
    
    lat = ds.createDimension('lat', mod.p.ylen)
    lon = ds.createDimension('lon', mod.p.xlen)

    times = ds.createVariable('time', 'f4', ('time',))
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    value = ds.createVariable('value', 'f4', ('time', 'lat', 'lon',))
    value.units = 'ba_fraction'

    lats[:] = np.arange(-90, 90, 1.25)
    lons[:] = np.arange(-180, 180, 1.875)
    
    if annual == True:
    
        value[:, :, :] = vals
        
    else:
        
        value[:, :, :] = np.stack(vals)

    ds.close()


