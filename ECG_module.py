#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:59:32 2020

@author: anamayshetty
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

class ECG:
            
    def __init__(self, heart_structure):
        self.struct = heart_structure
        self.electrical_activity = np.zeros(heart_structure.shape)
        self.refractory_cells = np.zeros(heart_structure.shape)
        
    def viz_collapsed_whole(self, image):
        "Visualises the 3-D structure with colour representin level - only used for Booleans"
        collapsed_grid = np.zeros((self.struct.shape[0:2]))
        
        if image == "active":
            for i in range(self.struct.shape[2]):
                collapsed_grid += i * self.electrical_activity[:, :, i]
        elif image == "structure":
            for i in range(self.struct.shape[2]):
                collapsed_grid += i * self.struct[:, :, i]
        else:
            print("Error - Invalid selection")
            
        plt.imshow(collapsed_grid)
        plt.colorbar()
        
    def viz_slice(self, image, z_level):
        "Visualises a single z-slice of the heart"
        if image == "active":
            plt.imshow(self.electrical_activity[:, :, z_level] + 0)
        elif image == "refractory":
            plt.imshow(self.refractory_cells[:, :, z_level] + 0)
        elif image == "structure":
            plt.imshow(self.struct[:, :, z_level] + 0)   
        else:
            print("Error - Invalid selection")
        plt.colorbar()
    
    def reset(self, x, y, z):
        "This resets the electrical activity, and starts the electrical wave"
        self.electrical_activity[:, :, :] = 0
        self.refractory_cells[:, :, :] = 0
        self.electrical_activity[x, y, z] = 1 
        
    
    def propogate(self):
        "Propogates the electrical wave forward by one step"
        refractory_period = 120
        # We first propogate the electrical signal in all directions
        electrical_prop = []
        for x, y in itertools.product([-1, 1], [0, 1, 2]):
            # We multiply by mask to identify where the electrical activity
            # can actally spread in the atria
            electrical_prop.append(
                    np.multiply(np.roll(self.electrical_activity, x, axis = y), self.struct)
                    )
        self.refractory_cells += refractory_period * np.minimum(self.electrical_activity, 1)
        # We then reduce all current activity by one
        # and ensure the unstimulated does not go to -1
        self.refractory_cells = np.maximum(self.refractory_cells - 1, 0)
        self.previous_active_cells = self.electrical_activity
        
        self.electrical_activity = (
                np.sum(electrical_prop, axis = 0) - self.refractory_cells
                ) >= 1  
    
    def calculate_norm_vector(self):
        # This will get the centre of the depolarise cells and +ve values
        positive_centre = np.average(
                np.where(self.refractory_cells >= 1), 
                axis = 1
                )
        # This will get the centre of the resting cells and -ve values
        negative_centre = np.average(
                np.where(self.struct - np.minimum(self.refractory_cells, 1) == 1), 
                axis = 1
                )
        # This generates the electrical vector
        electrical_vector = negative_centre - positive_centre
        # This returns the normalised vector
        return(electrical_vector/np.sqrt(np.sum(electrical_vector**2)))