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


def predict_from_tree(dat, tree, struct, prob = 'yprob.TRUE', 
                      na_return = 0, skip_val = -3.3999999521443642e+38):
    
    ''' 
    
    data to predict on
    tree frame from R
    structure object from define_tree_links
    
       
    prob should be 'yprob.TRUE' for class
    prob should be 'yval' for regression
    
    na_return gives the value to be returned where data are missing
    skip_val gives a value for nc.Dataset.fill_value - defaults to R default val
    
    '''
        
    
    if any(dat == skip_val):
        
        return(na_return)
    
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



def update_pars(tree, thresholds, probs, method = str, 
                target = 'yprob.TRUE', source = 'TRUE.', boot_int = 0):
       
    '''
    Updates a tree frame from a set of bootstrapped parameters
    boot_int can be given as a key-word arg to move sequentially through the parameter list
    
    '''
    
    target_col = np.where(tree.columns == target)[0][0]
    source_col = np.where(probs[0].columns == source)[0][0]
    
    thresh_n = 0
    prob_n   = 0
    
    if method == 'random':
    
        r = np.random.randint(thresholds[0].shape[0])  
    
        for i in range(tree.shape[0]):
        
            if tree['var'][i] != '<leaf>':
            
                tree.iloc[i, 5] = str('<' + str(thresholds[thresh_n].iloc[r, 0]))
                thresh_n += 1
                
            elif tree['var'][i] == '<leaf>':
                
                tree.iloc[i, target_col] = probs[prob_n].iloc[r, source_col]
                prob_n += 1
        
    elif method == 'bootstrapped':
    
        for i in range(tree.shape[0]):
        
            if tree['var'][i] != '<leaf>':
            
                tree.iloc[i, 5] = str('<' + str(thresholds[thresh_n].iloc[boot_int, 0]))
                thresh_n += 1
                
            elif tree['var'][i] == '<leaf>':
                
                tree.iloc[i, target_col] = probs[prob_n].iloc[boot_int, source_col]
                prob_n += 1
    
    return(tree)
    


#######################################################################

### Try to make predict function faster

#######################################################################

def predict_from_tree_fast(dat, tree, struct, prob = 'yprob.TRUE', 
                      na_return = 0, skip_val = -3.3999999521443642e+38):

    
    dat['missing'] = dat.apply(lambda y: skip_val in [float(x) for x in y], axis = 1)
    
    ncol           = dat.shape[1]
    
    for i in range(tree.shape[0]):
        
        if tree.iloc[i, 1] != '<leaf>':
            
            dat[tree.iloc[i, 1] + '_' + str(i)] = dat[tree.iloc[i, 1]] < float(tree.iloc[i, 5][1:])
            
            
        elif tree.iloc[i, 1] == '<leaf>':
        
           
            dat[tree.iloc[i, 1] + '_' + str(i)] = tree[prob].iloc[i]
    
    
    ### go from columns to predictions - needs a third column
    dat['Next_node']       = dat.iloc[:, ncol] #boolean - determines direction after next split
    dat['Destination']     = 0 #number of nodes / leaf in structure dictionary object
    dat['Probability_out'] = np.nan #captures output probs - runs until all filled
    
    ### Run prediction
    while len(np.where(np.isnan(dat['Probability_out']) == True)[0]) > 0:
        
        ### how to look for nodes that have found their prob?
        left = [struct[z]['Dest'][0] for x, y, z in zip(dat['Next_node'], dat['Probability_out'], dat['Destination']) if x == True and np.isnan(y)]
        right= [struct[z]['Dest'][1] for x, y, z in zip(dat['Next_node'], dat['Probability_out'], dat['Destination']) if x == False  and np.isnan(y)]
        
        ### update destination
        dat.loc[np.logical_and(dat['Next_node'] == True, np.isnan(dat['Probability_out'])), 'Destination']  = left
        dat.loc[np.logical_and(dat['Next_node'] == False, np.isnan(dat['Probability_out'])), 'Destination'] = right
        
        ### update output probs
        out_bool  = [struct[x]['Type'] == '<leaf>' for x in dat['Destination']]
        out_probs = [tree.loc[:, prob].iloc[int(x)] for x in dat['Destination'] if struct[int(x)]['Type'] == '<leaf>']
        dat.loc[out_bool, 'Probability_out'] = out_probs

        ### update Next node
        Node_update                                            = [ncol+x for x, y in zip(dat['Destination'], dat['Probability_out']) if np.isnan(y)]
        Node_update_key                                        = [x for x in np.where(np.isnan(dat['Probability_out']))[0]]
        dat.loc[np.isnan(dat['Probability_out']), 'Next_node'] = [dat.iloc[int(x), int(y)] for x, y in zip(Node_update_key, Node_update)]
        
               
    dat.loc[dat['missing'] == True, 'Probability_out'] = na_return
    
    res = dat.loc[:, 'Probability_out']
    dat = dat.iloc[:, 0:(ncol+1)]
    
    return(res)
        



