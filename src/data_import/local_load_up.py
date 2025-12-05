# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:16:40 2021

@author: Oli
"""

import os
from copy import deepcopy
import re

import pandas as pd
import numpy as np
import netCDF4 as nc

from data_import.load_funs import mk_par_dict


def load_local_data(root=None, map_folder=None, Dist=True, Maps=True, Fire=True, Nfer=True):
    """Load all local parameter and map data and return (Core_pars, Map_data).

    Args:
        root (str | None): root directory containing parameter files. If None,
            a repository-local default is used (useful for tests).
        map_folder (str | None): directory containing map/netCDF files. If None,
            a default relative to `root` is used.
        Dist, Maps, Fire, Nfer (bool): switches controlling which data blocks to load.

    Returns:
        tuple: (Core_pars, Map_data)

    Notes:
        - This function follows the behaviour of the original `local_load_up.py`.
        - It constructs `Core_pars` and `Map_data` and performs the same
          auxiliary computations as the original module.
    """

    # sensible defaults: use repository-local paths when not provided
    if root is None:
        root = r'C:/Users/olive/Documents/WHAM_Postdoc/WHAM/v1_025/pars/'
    if map_folder is None:
        map_folder = r'C:/Users/olive/Documents/WHAM_Postdoc/WHAM/v1_025/drive/'

    # normalize paths
    root = str(root).replace('\\', '/')
    map_folder = str(map_folder).replace('\\', '/')

    Rlen = len(root)
    Mlen = len(map_folder)

    # gather all files under the root
    file_list = []
    for path, _subdirs, files in os.walk(root):
        for name in files:
            file_list.append(os.path.join(path, name))
    file_list = [s.replace('\\', '/') for s in file_list]

    # empty dict to house files / parameters
    Core_pars = {
        'AFT_dist': '',
        'Dist_pars': {'Thresholds': '', 'Probabilities': ''},
        'Fire_use': {},
        'Fire_suppression': '',
        'Fire_escape': {},
        'Nfer_use': {}
    }

    # ----- Distribution parameters -----
    if Dist:
        AFT_dist = [s.replace('\\', '/') for s in file_list if '/Distribution' in s]

        Core_pars['AFT_dist'] = mk_par_dict(dat=AFT_dist, filt='Tree_frame', name_key=[Rlen, 13, 15])

        Core_pars['Dist_pars']['Thresholds'] = mk_par_dict(
            dat=AFT_dist, filt='thresholds', kind='multiple', name_key=[Rlen, 13, 17]
        )

        Core_pars['Dist_pars']['Probs'] = mk_par_dict(
            dat=AFT_dist, filt='probs', kind='multiple', name_key=[Rlen, 13, 12]
        )

    # ----- Fire use parameters -----
    if Fire:
        Fire_pars = [s.replace('\\', '/') for s in file_list if 'Fire use' in s]

        Core_pars['Fire_use']['bool'] = mk_par_dict(dat=Fire_pars, filt='bool.csv', kind='single', name_key=[Rlen, 14, 9])
        Core_pars['Fire_use']['ba'] = mk_par_dict(dat=Fire_pars, filt='ba.csv', kind='single', name_key=[Rlen, 14, 7])

        Escape_pars = [s.replace('\\', '/') for s in file_list if 'Fire escape' in s]

        escape_dict = {}
        escape_dict['fire_types'] = mk_par_dict(dat=Escape_pars, filt='tree', kind='single', name_key=[Rlen, 17, 9])

        # find the pars csv in Escape_pars (same heuristic as original)
        escape_pars_csv = [s for s in Escape_pars if 'Fire_escape_pars' in s]
        if escape_pars_csv:
            escape_dict['Overall'] = pd.read_csv(escape_pars_csv[0])
        else:
            escape_dict['Overall'] = pd.DataFrame()

        Core_pars['Fire_escape'] = escape_dict

    # ----- Nfer use parameters -----
    if Nfer:
        Core_pars['Nfer_use']['tree'] = ''
        Core_pars['Nfer_use']['lm'] = ''

        Nfer_pars = [s.replace('\\', '/') for s in file_list if 'Nfer_use' in s]
        tree_pars = [s for s in Nfer_pars if 'tree.csv' in s]
        lm_pars = [s for s in Nfer_pars if 'lm.csv' in s]

        tree_dict = {}
        if tree_pars:
            tree_dict = dict(zip([x[(Rlen + 14):-9] for x in tree_pars], [pd.read_csv(x) for x in tree_pars]))

        lm_dict = {}
        if lm_pars:
            lm_dict = dict(zip([x[(Rlen + 12):-7] for x in lm_pars], [pd.read_csv(x) for x in lm_pars]))

        Core_pars['Nfer_use']['tree'] = tree_dict
        Core_pars['Nfer_use']['lm'] = lm_dict

    # ----- Maps -----
    if Maps:
        Map_list = []

        for path, subdirs, files in os.walk(map_folder):
            for name in files:
                Map_list.append(os.path.join(path, name))

        Map_files  = [s.replace('\\', '/') for s in Map_list if ".nc" in s]
        Mask       = [s.replace('\\', '/') for s in Map_list if "Mask.csv" in s]
        Area       = [s.replace('\\', '/') for s in Map_list if "Area.csv" in s]

        Map_data = dict(zip([x[Mlen:-3] for x in Map_files], 
                [nc.Dataset(map_folder + x[Mlen:-3] + '.nc') for x in Map_files]))

        var_key  = zip([x for x in Map_data.values()], 
                   [[x for x in y.variables.keys()][len(y.variables.keys()) -2 ] for y in Map_data.values()])

        Map_data = dict(zip([x for x in Map_data.keys()], 
                [x[y][:] for x, y in var_key]))

        shp      = Map_data[[x for x in Map_data.keys()][0]].shape[1] * Map_data[[x for x in Map_data.keys()][0]].shape[2]
        Map_data['Mask'] = np.array(pd.read_csv(Mask[0])).reshape(shp)
        Map_data['Area'] = np.array(pd.read_csv(Area[0])).reshape(shp)
        

        # auxiliaries / convolutions (keep same operations as original)
        try:
            Map_data['Market.Inf'] = Map_data['GDP'] * Map_data['MA']
            Map_data['HDI_GDP'] = np.log(Map_data['GDP']) * Map_data['HDI']
            Map_data['W_flat'] = (1 / Map_data['TRI']) * Map_data['GDP']
            Map_data['Veg'] = Map_data['LUH2_secdf'] + Map_data['LUH2_secdn'] + Map_data['Other_vegetation']
            Map_data['NPP_mountain'] = Map_data['NPP'] * Map_data['TRI']
            Map_data['NPP_lpop'] = Map_data['NPP'][0:27, :, :] * np.log(Map_data['Pop'])
            Map_data['ET_pop'] = Map_data['ET'][0:27, :, :] / np.select([Map_data['Pop'] >= 1], [Map_data['Pop']], default=1)

            # handle missing values in processed data
            if 'HDI_GDP' in Map_data and hasattr(Map_data['HDI_GDP'], 'shape'):
                for i in range(Map_data['HDI_GDP'].shape[0]):
                    # be defensive: some arrays may use .data attribute (e.g., netCDF variables)
                    try:
                        arr = Map_data['HDI_GDP'].data
                        arr[i, :, :] = np.select([~np.isnan(arr[i, :, :])], [arr[i, :, :]], default=-3.3999999521443642e38)
                    except Exception:
                        arr = Map_data['HDI_GDP']
                        arr[i, :, :] = np.select([~np.isnan(arr[i, :, :])], [arr[i, :, :]], default=-3.3999999521443642e38)

                    try:
                        wflat = Map_data['W_flat'].data
                        wflat[i, :, :] = np.select([wflat[i, :, :] != 0], [wflat[i, :, :]], default=-3.3999999521443642e38)
                    except Exception:
                        wflat = Map_data['W_flat']
                        wflat[i, :, :] = np.select([wflat[i, :, :] != 0], [wflat[i, :, :]], default=-3.3999999521443642e38)
        
        except Exception:
            # if any auxiliary calculation fails quit
            raise RuntimeError("Calculation of auxiliary variables failed")

    # explicit garbage collection (mirrors original end-of-file behaviour)
    import gc
    gc.collect()

    return Core_pars, Map_data


if __name__ == '__main__':
    # when run as a script, load with defaults and display summary
    Core_pars, Map_data = load_local_data()
    print('Loaded Core_pars keys:', list(Core_pars.keys()))
    print('Loaded Map_data keys:', list(Map_data.keys()))

