# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:41:34 2024

@author: Oli
"""

MOSH_fire = []
SOSH_fire = []
Int_ar    = []

x = 'crb'

for y in range(26):
    
    mod.timestep = y
    mod.agents[1].fire_use()
    SOSH_fire.append(mod.agents[1].Fire_vals['crb'])
    
    mod.agents[2].fire_use()
    MOSH_fire.append(mod.agents[2].Fire_vals['crb'])

    mod.agents[3].fire_use()
    Int_ar.append(mod.agents[3].Fire_vals['crb'])
    
    


