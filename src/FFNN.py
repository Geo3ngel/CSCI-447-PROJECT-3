# ----------------------------------------------
# @file    ffnn.py
# @authors Dana Parker, Henry Soule, Troy Oster, George Engel

# ----------------------------------------------
# Standard-library imports
import random
import json

# ----------------------------------------------
# Third-party imports
import numpy as np

# ----------------------------------------------
# Custom imports
import Cost_Functions as cost

# ----------------------------------------------
# The Feed-Forward Neural Network object
class FFNN():
    
    # ----------------------------------------------
    # Constructor
    #
    # in_layer_sizes    Contains the number of neurons
    #                   in the i-th layer
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # Initializes weights via a normal distribution.
        self.weights = [np.random.randn(x, y)
            for y, x in zip(layer_sizes[:-1], layer_sizes[1:])]

        # Initializes biases via a normal distribution.
        #
        # We don't put a bias on the weights between
        # the input layer and the first layer,
        # hence layer_sizes[1:].
        self.biases = [np.random.randn(x, 1)
            for x in self.layer_sizes[1:]]
