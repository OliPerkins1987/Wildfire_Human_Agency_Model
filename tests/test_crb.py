# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:30:02 2021

@author: Oli
"""

import os 
import pandas as pd
import numpy as np
import pytest
import agentpy


os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')


os.chdir(str(wd + '/test_data/Fire').replace('\\', '/'))
SOSH    = pd.read_csv('SOSH.csv')
MOSH    = pd.read_csv('MOSH.csv')
Intense = pd.read_csv('Intense_arable.csv')

Py_SOSH = pd.read_csv('PySOSH.csv')
Py_MOSH = pd.read_csv('PyMOSH.csv')
Py_Intense = pd.read_csv('PyIntense_arable.csv')

def test_SOSH():
    
    v      = Py_SOSH.Probability_out.value_counts()
    
    SOSH_error = SOSH.x - Py_SOSH.Probability_out
    SOSH_error = [x for x in SOSH_error if x != 21.18421053]
    SOSH_error = [x for x in SOSH_error if np.abs(x) > 0.0001]
    
    
    errors = []
    
    
    if not [x for x in v.values] == [18682, 5318, 3648]:
        errors.append("Error in SOSH Prediction")
    
    if not len(SOSH_error) < 30:
        errors.append("Error in SOSH Prediction")
        
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_MOSH():
    
    assert(np.array_equal(MOSH['x'].value_counts().values, 
                          Py_MOSH.Outprobs.value_counts().values))


def test_IA():
    
    assert(np.array_equal(Py_Intense['0'].value_counts().values, 
                          np.array([18682,  6868,  2098])))
    


