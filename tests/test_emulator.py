# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:05:09 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

os.chdir((wd[0:-6] + '/src/data_import'))
exec(open("local_load_up.py").read())
exec(open("load_INFERNO_offline.py").read())


os.chdir(str(wd + '/test_data/R_outputs').replace('\\', '/'))
R_fire = pd.read_csv('Fire_1990.csv')


from model_interface.wham import WHAM
from Core_functionality.AFTs.agent_class import AFT

from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p
from Core_functionality.AFTs.forestry_afts  import Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts  import Hunter_gatherer, Recreationalist, SLM, Conservationist

from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

from Core_functionality.top_down_processes.arson import arson
from Core_functionality.top_down_processes.background_ignitions import background_rate
from Core_functionality.top_down_processes.fire_constraints import fuel_ct, dominant_afr_ct
from Core_functionality.top_down_processes.fire_control_measures import fire_control_measures
from Core_functionality.top_down_processes.deforestation import deforestation
from Core_functionality.top_down_processes.fire_suppression import fire_fighter

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation

from modis_emulator.mod_em import modis_em 

#####################################################################

### Run model year then reproduce outputs

#####################################################################

all_afts = [Swidden, SOSH, MOSH, Intense_arable, 
            Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p,
            Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer, Recreationalist, SLM, Conservationist]

parameters = {
    
    ### Model run limits
    'xlen': 192, 
    'ylen': 144,
    'start_run': 0,
    'end_run' : 0,
    
    ### Agents
    'AFTs': all_afts,
    
    'LS'  : [Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex],
    
    'Observers': {'background_rate': background_rate, 
                  'arson': arson, 
                  'fuel_constraint': fuel_ct, 
                  'dominant_afr_constraint': dominant_afr_ct, 
                  'fire_control_measures': fire_control_measures, 
                  'deforestation': deforestation, 
                  'fire_suppression': fire_fighter},    
    
    'Fire_seasonality': Seasonality,
    
    ### AFT distribution parameter
    'theta'   : 0.1,
    
    ### data
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    
    ### Fire parameters
    'Fire_types': {'cfp': 'Vegetation', 'crb': 'Arable', 'hg': 'Vegetation', 
                   'pasture': 'Pasture', 'pyrome': 'Vegetation', 'defor': 'Vegetation'}, 

    ### constraints
    'Constraint_pars': {'Soil_threshold': 0.1325, 
                        'Dominant_afr_threshold': 0.5, 
                        'Rangeland_stocking_contstraint': True, 
                        'R_s_c_Positive' : False, 
                        'HG_Market_constraint': 7800, 
                        'Arson_threshold': 0.5},
    
    ### Deforestation fire fraction
    'Defor_pars': {'Pre'    : 1, 
                   'Trans'  : 0.84, 
                   'Intense': 0.31},
    
    
    ### fire meta pars
    'Seasonality'  : False, 
    'escaped_fire' : False,
    
    ### MODIS emulation
    'emulation'    : False, ## run incrementally after single WHAM run

    ### reporters
    'reporters': ['Managed_fire', 'Background_ignitions', 'Arson'],
    
    ### house keeping
    'bootstrap': False,
    'n_cores'  : 4,
        
    'write_annual': False,
    'write_fp': r'C:\Users\Oli\Documents\PhD\wham\results'  
    
    }


mod = WHAM(parameters)

### setup
mod.setup()

### ignite
mod.go()

#######################################################################################

### tests

#######################################################################################

em = modis_em(mod, dgvm = False)

def test_em_getter():
    
    errors = []
    
    em.get_fire_vals()
    
    faulty_val = [x for y in (em.Managed_fire['crb'].values()) for x in y if abs(x) > 1 and x > 0]
    
    if not len(faulty_val) == 0:
        
        errors.append("Incorrect gathering of fire data")
    
    if not np.array([x for x in em.Managed_fire['crb'].values()]).shape == (3, 27648):

        errors.append("Incorrect gathering of fire data")
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
        
def test_em_cell_division():
    
    errors = []
    em.divide_cells()
    
    
    ### mask == actual?
    if not np.where(em.LC_pix['Mask'] > 0)[0][0] == 1012:
        
        errors.append("Land mask incorrectly imported")
    
    
    ### land cover data = actual?
    lc_pix = dict(zip([x for x in em.LC_pix.keys()], 
                       [x[0] for x in em.LC_pix.values()]))

    if not lc_pix['Arable'] == 0:
        
        errors.append('Land cover data incorrectly imported')
        
    
    ### check cell divions
    lc_pix = dict(zip([x for x in em.LC_pix.keys()], 
                       [x[1012] for x in em.LC_pix.values()]))

    tot_pix= round(lc_pix['n_MODIS'] * lc_pix['Mask']) 

    if not lc_pix['Vegetation'] + lc_pix['Pasture'] + lc_pix['Arable'] == tot_pix:
        
        errors.append('Cell division incorrect')

        
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_aft_assignment():
    
    cn     = 17016
    errors = []
    fire_res = em.assign_managed_fires(cn) #max BA from emulator
    
    ################################
    ### ? n_cells = total
    ################################
    
    lc_pix = dict(zip([x for x in em.LC_pix.keys()], 
                       [x[cn] for x in em.LC_pix.values()]))
    
    lc_pix = [lc_pix['Arable'], lc_pix['Pasture'], lc_pix['Vegetation']]
    
    if not [x.shape[0] for x in fire_res.values()] == lc_pix:
    
        errors.append('Incorrect number of possible fire cells by land sytem')

    
    ###################################
    ### Distribution of fires in cells?
    ###################################
    
    a_area = []
    
    for a in em.abm.agents:   
    
        a_area.append(em.get_AFT_area(a, cn))

    abm_area = [x.reshape(144*192)[cn] for x in em.abm.X_axis.values()]
    abm_area = [abm_area[x] for x in [0, 1, 2, 3, 6]]
    
    if not round(sum(a_area), 3) == round(sum(abm_area), 3):
        
        errors.append('AFT distribution incorrectly imported')
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_managed_fire_assignment():
    
    cn     = 17016 #max BA from emulator
    errors = [] 
    
    a = [x for x in em.abm.agents][5]
    i = 'pasture'
    lc= 'Pasture'
    AFT_area = em.get_AFT_area(a, cn)

    #######################################
    ### rate of poisson distribution       
    #######################################
                     
    fire_rate = em.get_rate(em.Managed_igs[i][a][cn],
                               em.LC_pix['n_MODIS'][cn], 
                                   AFT_area)
    
    if not round(fire_rate, 3) == 0.858:
        
        errors.append('Fire rate calculated incorrectly')
        
    #####################################
    ### poisson distribution of fires
    #####################################
    
    fire_dist = em.poisson_fires(
        fire_rate, em.LC_pix[lc][cn], AFT_area, cn)
    
    pred_outcome = fire_rate * (em.LC_pix['n_MODIS'][cn] / em.LC_pix['Pasture'][cn]) * AFT_area
    
    ### does mean of predictions ~ rate * occupancy fraction?
    if not round(pred_outcome, 3) == round(np.nanmean(fire_dist), 3):
       
       errors.append('Distribution of fires conducted incorrectly')
    
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_fire_distribution():
    
    tmp    = em.assign_managed_fires(17016)
    BA     = em.distribute_ba(tmp)
    filtBA = em.filter_ba_MODIS(BA)
    errors = []
    
    if not np.nanmax(BA['Pasture']) <= 21.0:
        
        errors.append('Fire allocation did not work')
    
    if not np.nanmean(tmp['Pasture']) == np.nanmean(BA['Pasture']):
        
        errors.append('Fire smoothing did not work')
    
    BA['Pasture'] = np.array([x if x >= 21 else 0 for x in BA['Pasture']])
    
    if not np.nanmean(filtBA['Pasture']) == np.nanmean(BA['Pasture']):
        
        errors.append('MODIS smoothing did not work')
        
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
        
    
################################################################

### tests for INFERNO emulation

################################################################

em = modis_em(mod, dgvm = INFERNO)

def test_dgvm_load():
    
    em.setup_emulate()
    
    errors = []
    
    ### check loading of fire data
    if not np.nanmean(em.Lightning['igs'].reshape(144*192)) == np.nanmean(
            (em.dgvm['Lightning_fires'].data * em.abm.Area * (1-em.abm.Suppression)).reshape(144*192)):
        
        errors.append('Calculation of fire suppression incorrect')
    
    if not np.nanmean(em.Bare_soil['Arable']) < np.nanmean(em.abm.X_axis['Cropland']):
        
        errors.append('Arable bare soil fraction loaded incorrectly')
                
    if not np.nanmean(em.Bare_soil['Pasture']) < np.nanmean(em.abm.X_axis['Rangeland'] + em.abm.X_axis['Pasture']):
        
        errors.append('Arable bare soil fraction loaded incorrectly')
        
        
        
        

        
        