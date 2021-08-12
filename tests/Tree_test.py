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

from Core_functionality.Trees.Transfer_tree import define_tree_links, predict_from_tree, predict_from_tree_fast
exec(open("test_setup.py").read())

###############

### Load test data

###############

os.chdir(str(test_dat_path) + '\Trees')

tree_frame = pd.read_csv('Swidden_tree.csv')
tree_dat   = pd.read_csv('Swidden_dat.csv')

HG_dat     = pd.read_csv('HG_dat.csv')
HG_tree    = pd.read_csv('HG_tree.csv')

Residue_dat  = pd.read_csv('Residue_data.csv')
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
    preds       = tree_dat.apply(predict_from_tree, 
                   axis = 1, tree = tree_frame, struct = tree_struct, prob = 'yprob.TRUE')
    
    assert(preds.to_list() == pytest.approx(tree_dat['overall.pred'].to_list(), abs = 1e-2))


def test_tree_reg_pred():
    
    tree_struct = define_tree_links(Residue_tree)
    preds       = Residue_dat.apply(predict_from_tree, 
                   axis = 1, tree = Residue_tree, struct = tree_struct, prob = 'yval')
    
    
    R_preds     = Residue_dat['pred']
    assert(preds.to_list() == pytest.approx(R_preds.to_list(), abs = 1e-2))



### prediction tests

def test_tree_class_pred_fast():
    
    tree_struct = define_tree_links(tree_frame)
    preds       = predict_from_tree_fast(dat = tree_dat, tree = tree_frame, 
                                    struct = tree_struct, prob = 'yprob.TRUE')
    
    assert(preds.to_list() == pytest.approx(tree_dat['overall.pred'].to_list(), abs = 1e-2))


def test_tree_reg_pred_fast():
    
    tree_struct = define_tree_links(Residue_tree)
    preds       = predict_from_tree_fast(dat = Residue_dat,
                     tree = Residue_tree, struct = tree_struct, prob = 'yval')
    
    
    R_preds     = Residue_dat['pred']
    assert(preds.to_list() == pytest.approx(R_preds.to_list(), abs = 1e-2))

