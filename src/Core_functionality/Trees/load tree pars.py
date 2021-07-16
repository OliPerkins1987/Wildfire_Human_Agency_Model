# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:21:25 2021

@author: Oli
"""


os.chdir(r'F:\PhD\Model files\Distribution\Trees\Cropland\Pre')

files = [x for x in os.listdir() if '.csv' in x]

p  = [pd.read_csv(x) for x in files if 'Probs' in x]
wp = [pd.read_csv(x) for x in files if 'Weighted_probs' in x]

t  = [pd.read_csv(x) for x in files if 'Thresholds' in x]
wt = [pd.read_csv(x) for x in files if 'Weighted_thresholds' in x]


probs      = {'p':p, 'wp':wp}
thresholds = {'t':t, 'wt':wt}
tree = pd.read_csv('Tree_frame.csv')



