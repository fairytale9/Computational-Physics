#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random
import tensorflow as tf


n = 64  # size of the net 64*64
training_images = np.zeros((10000, n, n))
training_labels = np.zeros(10000)

for num in range(10000):
    
    # Using Monte Carlo method to generate models
    Net = np.zeros((n+1, n+1))
    p = np.random.uniform(0, 1)
    for row in range(n):
        for column in range(1, n+1):
            rn = random.random()
            if rn < p:
                Net[row, column] = 1
            else:
                continue
    training_images[num, :, :] = Net[0:n, 1:n+1] 
    
    # Using the Hoshen-Kopelman algorithm to label the model
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
    training_labels[num] = label


# visualize the two-dimensional percolation model
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[1])
print(training_labels[1])
print(training_images[1])


# training the neural network
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(128, activation=tf.nn.relu),
                  tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

