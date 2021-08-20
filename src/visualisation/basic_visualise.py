# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:03:42 2021

@author: Oli
"""

import matplotlib.pyplot as plt

def map_output(data):

    for image in data:    

        plt.imshow(image, cmap='plasma')
        plt.colorbar()
        plt.show()
    

    
