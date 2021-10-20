#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random


# test the machine learning model
accuracy = np.zeros(19)

for loop in range(19):
    p = 0.05 * (loop + 1)
    
    # generating test data and labels
    n = 64
    test_images = np.zeros((10000, n, n))
    test_labels = np.zeros(10000)
    for num in range(10000):
        Net = np.zeros((n+1, n+1))
        for row in range(n):
            for column in range(1, n+1):
                rn = random.random()
                if rn < p:
                    Net[row, column] = 1
                else:
                    continue
        test_images[num, :, :] = Net[0:n, 1:n+1] 
        Net_copy = Net
        index = 1
        for row in range(n-1, -1, -1):
            for column in range(1, n+1):
                if Net_copy[row, column] == 1:
                    if Net_copy[row, column-1]+Net_copy[row+1, column] == 0:
                        Net_copy[row, column] = index
                        index += 1
                    elif Net_copy[row, column-1] == 0:
                        Net_copy[row, column] = Net_copy[row+1, column]
                    elif Net_copy[row+1, column] == 0:
                        Net_copy[row, column] = Net_copy[row, column-1]
                    else:
                        m = min(Net_copy[row, column-1], Net_copy[row+1, column])
                        M = max(Net_copy[row, column-1], Net_copy[row+1, column])
                        Net_copy[Net_copy == M] = m
                        Net_copy[row, column] = m
          
        first_line = Net_copy[0, 1:n+1]
        first_line = [x for x in first_line if x != 0]
        last_line = Net_copy[n-1, 1:n+1]
        last_line = [x for x in last_line if x != 0]
        for i in first_line:
            if i in last_line:
                label = 1
                break
            label = 0
        test_labels[num] = label
    eval = model.evaluate(test_images, test_labels)
    accuracy[loop] = eval[1]

print(accuracy)
