# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:46:04 2021

@author: Oli
"""

import pandas as pd
import numpy as np
import os
import pytest

#######################################################################

### Takes an R tree frame and returns a dictionary to predict from

#######################################################################

def define_tree_links(tree):
    
    tree_struct = {}
    
    for i in range(len(tree.iloc[:, 1])):
        
        if(tree.iloc[i, 1] != '<leaf>'):
            
            tree_struct[i] = {}
            tree_struct[i]['Type'] = 'Node'
    
            tree_struct[i]['Dest']         = []
            
            current_node                   = tree.iloc[i, 0]
            
            ##########################################################
            
            ### Check this works with robust testing!
            
            ##########################################################
            
            next_node                      = [i+1]
            
            if(i == 0):
            
                next_node.append([int(x) for x in (np.where(tree.iloc[:,0] == current_node + 2))][0])
            
            elif(tree.iloc[next_node[0], 1] == '<leaf>'):
                
                next_node.append(i+2)
            
            else:
                
                next_node.append([int(x) for x in (np.where(tree.iloc[:,0] == (tree.iloc[next_node[0],0]+1)))][0])
                    
            
            tree_struct[i]['Dest']         = next_node
    
        
        else:
            
            tree_struct[i] = {'Type': '<leaf>'}
    

    return(tree_struct)


def predict_from_tree(dat, tree, struct, prob = 'yprob.TRUE'):
    
    ''' 
    
    data to predict on
    tree frame from R
    structure object from define_tree_links
    
       
    prob should be 'yprob.TRUE' for class
    prob should be 'yval' for reg
    
    '''
    
    tree_key = 0
    tree_type= 'Node'
    
    while(tree_type == 'Node'):
    
        var = tree.iloc[tree_key, 1]
    
        if(dat[var] < float(tree.iloc[tree_key, 5][1:])):
            
            tree_key  = struct[tree_key]['Dest'][0]
            tree_type = struct[tree_key]['Type']
        
        else:
            
            tree_key  = struct[tree_key]['Dest'][1]
            tree_type = struct[tree_key]['Type']
            
    
    pred = tree[prob][tree_key]
    
    return(pred)


