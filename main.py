#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 23:02:12 2020

@author: anamayshetty
"""
import numpy as np
import matplotlib.pyplot as plt

from ECG_module import ECG

# This imports the previously produced heart scaffold
heart_scaffold = np.load("heart_scaffold.npy")

current_ecg = ECG(heart_scaffold)
current_ecg.reset(75, 90, 30)


lead_1_container = []
i = 0
while (np.sum(current_ecg.electrical_activity) > 0):
    current_ecg.propogate()
    
    normalised_vector = current_ecg.calculate_norm_vector()
    
    lead_1_container.append(np.dot(normalised_vector, [1, 0, 0]))
    
    if (i % 10 == 0):
        plt.title(i)
        current_ecg.viz_collapsed_whole("active")
        plt.show()
    
    i += 1
    
plt.plot(lead_1_container)
#plt.imshow(current_ecg.refractory_cells[:, :, 0] + 0)
#plt.plot(
#        [0, current_ecg.calculate_norm_vector()[0]], 
#        [0,current_ecg.calculate_norm_vector()[1]]
#        )
#plt.show()