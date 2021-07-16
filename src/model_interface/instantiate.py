# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:38:02 2021

@author: Oli
"""

#################################################

### Instantiate

#################################################

parameters = {
    
    'xlen': 192, 
    'ylen': 144,
    'AFTs': [Conservationist],
    'LS'  : [Unoccupied],
    'AFT_pars': Core_pars,
    'Maps'    : Map_data,
    'timestep': 0,
    'theta'    : 0.1,
    'bootstrap': True
    
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



