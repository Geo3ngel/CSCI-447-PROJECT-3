# ----------------------------------------------
# @file    ffnn.py
# @authors Dana Parker, Henry Soule, Troy Oster, George Engel

# ----------------------------------------------
# Standard-library imports
import random
import json
import math

# ----------------------------------------------
# Third-party imports
import numpy as np

# ----------------------------------------------
# Custom imports
import cost_functions as cost

# ----------------------------------------------
# The Feed-Forward Neural Network object
class FFNN():
    
    # ----------------------------------------------
    # Constructor
    #
    # in_layer_sizes    Contains the number of neurons
    #                   in the i-th layer
    def __init__(self, layer_sizes, db_type, 
        data, desired_out_col,
        num_epochs=50, using_test_data=False):

        self.layer_sizes = layer_sizes
        self.epochs = [[] for x in range(num_epochs)]
        self.db_type = db_type

        # Initializes weights via a normal distribution.
        self.weight_vec = [np.random.randn(x, y)
            for y, x in zip(layer_sizes[:-1], layer_sizes[1:])]

        # Initializes biases via a normal distribution.
        #
        # We don't put a bias on the weights between
        # the input layer and the first layer,
        # hence layer_sizes[1:].
        self.bias_vec = [np.random.randn(x, 1)
            for x in self.layer_sizes[1:]]

        if using_test_data is True:
            self.test_data, self.tr_data = \
                self.split_and_augment_data(data, desired_out_col)
        else:
            self.tr_data = self.augment_data(data, desired_out_col)


    # ----------------------------------------------
    # The counterpart to augment_data().
    # This function is for when we want to test the neural net using 
    # test / training data. (To do this should be much slower.)
    
    def split_and_augment_data(self, data, desired_out_col):
        random.shuffle(data)
 
        temp = []
        for ex in data:
            temp.append(ex, ex[desired_out_col])

        test_data = temp[0 : math.ceil(len(data) / 10)]
        tr_data = temp[math.ceil(len(data) / 10) + 1 : len(data)]

        return test_data, tr_data
    
    # ----------------------------------------------
    # The counterpart to split_and_augment_data().
    # This function is for when we want to only train the neural network,
    # which should be faster than if we were to also test the 
    # trained neural network.

    def augment_data(self, data, desired_out_col):
        return [(ex, ex[desired_out_col]) for ex in data]
    
    # ----------------------------------------------
    # Returns the output layer produced from `in_act`
    #
    # in_act    An activation vector of some layer
    def feed_forward(self, in_act_vec):
        for bias, weight in zip(self.weight_vec, self.bias_vec):
            out_act_vec = sig(np.dot(in_act_vec, weight) + bias)
        return out_act_vec
    
    # ----------------------------------------------
    # Trains the neural network via stochastic gradient descent
    # with mini-batches.
    #
    # tr_data      The training data
    # test_data    The test data
    # len_batch    The length of each "mini-batch"

    def grad_desc(self, tr_data, learning_rate,
        print_partial_progress=False, len_batch=20, test_data=None):

        # Variables to make the code cleaner
        if test_data:
            n_test = len(test_data)
        
        n_train = len(tr_data)
        n_epoch = len(self.epochs)

        # The gradient descent itself for every epoch
        for e in range(n_epoch):
            
            # Split the data into mini-batches
            random.shuffle(tr_data)
            batches = [tr_data[x : x + len_batch]
                for x in range(0, n_train, len_batch)]
            
            # For every mini-batch,
            # update the entire networks's weights and biases
            # via gradient descent and backpropagation
            for curr_batch in batches:
                new_bias = [np.zeros(b.shape)
                    for b in self.bias_vec]
                new_weight = [np.zeros(w.shape)
                    for w in self.weight_vec]

                for ex, desired_out in curr_batch:
                    delta_b, delta_w = self.back_prop(ex, desired_out)
                    new_bias = []
    
    # ----------------------------------------------
    # Basically finding the partial derivatives of
    # the cost with respect to both the weights and the biases

    def back_prop(self, activation, desired_out):
        new_bias = [np.zeros(b.shape)
            for b in self.bias_vec]
        new_weight = [np.zeros(w.shape)
            for w in self.weight_vec]
        act_list = [activation]
        sum_list = []

        for b, w in zip(self.bias_vec, self.weight_vec):
            curr_act = np.dot(w, curr_act) + b
            sum_list.append(curr_act)
            act_list.append(sig(curr_act))
        
        # Set initial values for the new bias
        new_bias = sig_prime(sum_list[-1]) \
            * self.cost_prime(act_list[-1])

        # We transpose because we need the list to be a column vector
        new_weight = np.dot(new_bias, act_list[-2].transpose())

        for layer_num in range(2, len(self.layer_sizes)):

            # Start at
            curr_sum = sum_list[-layer_num]
        

    # ----------------------------------------------
    # The derivative of our cost function
    # if the cost function is (a - y)^2

    # TODO: see if (out_acts-desired_out) is better
    def cost_prime(self, out_acts, desired_out):
        return 2*(out_acts-desired_out)            

    
# ----------------------------------------------
# w_dot_a_plus_b   A weighted sum which equals
#                  the dot product of the weight vector (w)
#                  and activation vector (a)
#                  plus the bias vector (b)
# 
# Vectorizes sigmoid over 
# Sigmoid in math notation: https://www.oreilly.com/library/view/hands-on-automated-machine/9781788629898/assets/1e81ffa6-1c5a-4d48-b88e-d01d4047f2ef.png

def sig(w_dot_a_plus_b):
    return 1.0 / (1.0 + np.exp(-w_dot_a_plus_b))

# ----------------------------------------------
# The derivative of the sigmoid function
def sig_prime(w_dot_a_plus_b):
    return sig(w_dot_a_plus_b) * (1 - sig(w_dot_a_plus_b))