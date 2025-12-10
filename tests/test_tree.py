# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:02:25 2021

@author: Oli
"""
import pytest 
import pandas as pd
import numpy as np
import os
import sys


os.chdir(os.path.dirname(os.path.realpath(__file__)))
wd = os.getcwd().replace('\\', '/')

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree_numpy


###############

### Load test data

###############

os.chdir(str(wd + '/test_data/Trees').replace('\\', '/'))

tree_frame = pd.read_csv('Swidden_tree.csv')
tree_dat   = pd.read_csv('Swidden_dat.csv')
tree_pred  = tree_dat['overall.pred']
tree_var   = [x for x in tree_dat.columns if x != 'overall.pred']
tree_dat   = np.array(tree_dat)

HG_dat     = pd.read_csv('HG_dat.csv')
HG_tree    = pd.read_csv('HG_tree.csv')

Residue_dat  = pd.read_csv('Residue_data.csv')
Residue_pred = Residue_dat['pred']
Residue_var  = [x for x in Residue_dat.columns if x != 'pred']
Residue_dat  = np.array(Residue_dat.iloc[:, [0, 0]])
Residue_tree = pd.read_csv('Residue_regression_tree.csv')

complex_tree_frame = pd.read_csv('Complex_Swidden_tree.csv')

  
### Structure tests

def test_tree_struct():
    
    tree_struct = define_tree_links(complex_tree_frame)
    Leaves      = np.where(np.array(
                        [tree_struct[x]['Type'] for x in [
                        y for y in tree_struct.keys()]]) == '<leaf>')
    
    True_Leaves = np.where(complex_tree_frame['var'] == '<leaf>')
    
    assert(np.array_equal(Leaves[0], True_Leaves[0]))
    

def test_tree_struct_2():
    
    tree_struct = define_tree_links(HG_tree)
    Leaves      = np.where(np.array(
                    [tree_struct[x]['Type'] for x in [
                    y for y in tree_struct.keys()]]) == '<leaf>')
    
    True_Leaves = np.where(HG_tree['var'] == '<leaf>')
    
    assert(np.array_equal(Leaves[0], True_Leaves[0]))
    

def test_tree_struct_3():
    
    tree_struct = define_tree_links(Residue_tree)
    Leaves      = np.where(np.array(
                    [tree_struct[x]['Type'] for x in [
                    y for y in tree_struct.keys()]]) == '<leaf>')
    
    True_Leaves = np.where(Residue_tree['var'] == '<leaf>')
    
    assert(np.array_equal(Leaves[0], True_Leaves[0]))
    

### prediction tests

def test_tree_class_pred():
    
    tree_struct = define_tree_links(tree_frame)
    preds       = predict_from_tree_numpy(dat = tree_dat, tree = tree_frame, 
                    split_vars = tree_var,struct = tree_struct, prob = 'yprob.TRUE')
    
    assert(preds == pytest.approx(tree_pred.to_list(), abs = 1e-2))


def test_tree_reg_pred():
    
    tree_struct = define_tree_links(Residue_tree)
    preds       = predict_from_tree_numpy(dat = Residue_dat,tree = Residue_tree, 
                 split_vars = ['GDP', 'GDP'], struct = tree_struct, prob = 'yval')
    
    assert(preds == pytest.approx(Residue_pred.to_list(), abs = 1e-2))

