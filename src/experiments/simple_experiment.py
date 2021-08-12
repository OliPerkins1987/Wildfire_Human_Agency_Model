# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:12:56 2021

@author: Oli
"""


#### Load
from model_interface.wham import WHAM
from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist, Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p
from Core_functionality.AFTs.forestry_afts  import Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts  import Hunter_gatherer, Recreationalist, SLM, Conservationist
from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex


#################################################

### Instantiate

#################################################

AFR_res = {}

for t in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
          0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]:

    parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [Swidden, SOSH, MOSH, Intense_arable, 
             Pastoralist, Ext_LF_r, Int_LF_r, 
             Ext_LF_p, Int_LF_p, 
             Agroforestry, Logger, Managed_forestry, Abandoned_forestry, 
             Hunter_gatherer, Recreationalist, SLM, Conservationist],
    'LS'  : [Cropland, Rangeland, Pasture, Forestry, Urban, Nonex, Unoccupied],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : t,
    'bootstrap': False
    
    }

    test = WHAM(parameters)
    
    ### setup
    test.setup()
    test.ls.setup()
    test.ls.get_pars(test.p.AFT_pars)
    test.ls.get_boot_vals(test.p.AFT_pars)
    test.agents.setup()
    test.agents.get_pars(test.p.AFT_pars)
    test.agents.get_boot_vals(test.p.AFT_pars)
    
    ### ls
    test.ls.get_vals()
    test.allocate_X_axis()
    
    ### AFT
    test.agents.compete()
    test.allocate_Y_axis()
    test.agents.sub_compete()
    test.allocate_AFT()

    AFR_res[str(t)] = test.LFS
    
    print(t)


####################################################

### output analysis

####################################################

res     = []

for i in range(len(AFR_res.values())):
    
    res.append(get_afr_vals([x for x in AFR_res.values()][i]))
    
    print('Teg')
    
    for j in res[i].keys():
        
        print('Cunt')
        
        res[i][j] = np.nanmean(res[i][j])

for c in range(len(['red', 'green', 'blue', 'yellow'])):
    
    plt.plot(afr_res.iloc[:, c],color=['red', 'green', 'blue', 'yellow'][c])
    
    


