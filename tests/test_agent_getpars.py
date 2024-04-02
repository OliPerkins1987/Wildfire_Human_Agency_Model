# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:25:15 2021

@author: Oli
"""

import pytest 
import pandas as pd
import numpy as np
import agentpy as ap
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

os.chdir((wd[0:-6] + '/src/data_import'))
exec(open("local_load_up.py").read())


from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree
from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.AFTs.arable_afts import SOSH, Intense_arable
from Core_functionality.AFTs.nonex_afts import Conservationist
from model_interface.wham import WHAM


#########################################################################

### Load test data

#########################################################################

os.chdir(str(wd + '/test_data/AFTs').replace('\\', '/'))
Cons_frame    = pd.read_csv('Conservationist_pars.csv')
Intense_frame = pd.read_csv('Intense_arable_pars.csv')
SOSH_frame    = pd.read_csv('SOSH_pars.csv')


##########################################################################

### tests

##########################################################################

@pytest.mark.usefixtures("mod_pars")
def test_SOSH_pars_load(mod_pars):
    
    errors = []
    
    mod        = WHAM(mod_pars)
    mod.agents = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.AFTs]])
    
    mod.agents.setup()
    mod.agents.get_dist_pars(mod.p.AFT_pars)
    
    if not (np.array_equal(mod.agents.Dist_frame[1]['yprob.TRUE'], SOSH_frame['yprob.TRUE'])):
        errors.append("AFT parameters not loaded correctly")
        
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
        
@pytest.mark.usefixtures("mod_pars")
def test_Conservationist_pars_load(mod_pars):
    
    errors = []
    
    mod        = WHAM(mod_pars)
    mod.agents = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.AFTs]])
    
    mod.agents.setup()
    mod.agents.get_dist_pars(mod.p.AFT_pars)
    
    if not (np.array_equal(mod.agents.Dist_frame[17]['yprob.TRUE'], 
                           Cons_frame['yprob.TRUE'])):
        errors.append("AFT parameters not loaded correctly")
        
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

@pytest.mark.usefixtures("mod_pars")
def test_Intense_arable_pars_load(mod_pars):
    
    errors = []
    
    mod        = WHAM(mod_pars)
    mod.agents = ap.AgentList(mod, 
                       [y[0] for y in [ap.AgentList(mod, 1, x) for x in mod.p.AFTs]])
    mod.agents.setup()
    mod.agents.get_dist_pars(mod.p.AFT_pars)
    
    if not (np.array_equal(mod.agents.Dist_frame[3]['yprob.TRUE'], 
                           Intense_frame['yprob.TRUE'])):
        errors.append("AFT parameters not loaded correctly")
        
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
