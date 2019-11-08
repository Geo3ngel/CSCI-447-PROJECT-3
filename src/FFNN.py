'''
----------------------------------------------
@file    ffnn.py
@authors Dana Parker, Henry Soule, Troy Oster, George Engel
'''

# ----------------------------------------------
# Standard-library imports
import random
import json
import math
import sys
from time import gmtime, strftime
from statistics import mean

# ----------------------------------------------
# Third-party imports
import numpy as np

# ----------------------------------------------
# Custom imports
import Cost_Functions as cost

# ----------------------------------------------
# The Feed-Forward Neural Network object
class FFNN():

    '''
    ----------------------------------------------
    Constructor

    in_layer_sizes    Contains the number of neurons
                      in the i-th layer
    '''
    def __init__(self, layer_sizes, db_type, db_name,
        data, learning_rate, num_epochs=200):

        # A list of integers representing
        # the number of nodes in layer i
        self.layer_sizes = layer_sizes
        self.num_epochs = num_epochs
        self.db_name = db_name
        self.db_type = db_type
        self.learning_rate = learning_rate
        self.data = data
        self.old_data = self.data[:]
        
        if db_type == 'classification':
            self.act_fn = sig
            self.act_fn_prime = sig_prime
        elif db_type == 'regression':
            self.act_fn = one_times
            self.act_fn_prime = one_times
        else:
            print('Invalid database type. Quitting.')
            sys.exit()

        # Initializes weights via a normal distribution.
        self.weight_vec = [np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

        # Initializes biases via a normal distribution.
        self.bias_vec = [np.random.randn(x, 1) for x in self.layer_sizes[1:]]

        print('-----------------------------------------------')
        print('learning rate: ' + str(self.learning_rate))
        print('layer sizes: ' + str(self.layer_sizes))
        self.grad_desc()

    

    '''
    ----------------------------------------------
    Returns the output layer produced from `in_act`
    
    in_act_vec    An activation vector of some layer
    '''
    def feed_forward(self, in_act_vec):
        for bias, weight in zip(self.bias_vec, self.weight_vec):
            in_act_vec = (self.act_fn)(np.dot(weight, in_act_vec) + bias)
        return in_act_vec

    '''
    ----------------------------------------------
    Trains the neural network via stochastic gradient descent
    with mini-batches.

    tr_data                 The training data
    test_data               The test data
    learning_rate           A real number between 0 and 1
    len_batch               The length of one "mini-batch"
    print_partial_progress  A boolean for whether we want to
                            print the evaluation of EVERY epoch
                            (WARNING: slow if True)
    '''
    def grad_desc(self, print_partial_progress=False):

        n_data = len(self.data)
        len_batch = math.ceil(n_data / 10)

        # The gradient descent itself for every epoch
        for e in range(self.num_epochs):

            # Randomly shuffle the training data
            random.shuffle(self.data)

            # Split the data into mini-batches
            batches = [self.data[x : x + len_batch]
                for x in range(0, n_data, len_batch)]

            # For every mini-batch,
            # update the entire networks's weights and biases
            # via one gradient descent iteration
            # (this step uses the back propagation)
            for curr_batch in batches:
                new_bias = [np.zeros(b.shape) for b in self.bias_vec]
                new_weight = [np.zeros(w.shape) for w in self.weight_vec]

                for ex, desired_out in curr_batch:
                    change_bias, change_weight = self.back_prop(ex, desired_out)
                    new_bias = [bias + change for bias, change in zip(new_bias, change_bias)]
                    new_weight = [weight + change for weight, change in zip(new_weight, change_weight)]

                # Apply momentum
                self.weight_vec = \
                    [w - (self.learning_rate / len_batch) * nw
                    for w, nw
                    in zip(self.weight_vec, new_weight)]

                self.bias_vec = \
                    [b - (self.learning_rate / len_batch) * nb
                    for b, nb
                    in zip(self.bias_vec, new_bias)]

            # Print results of the epochs
            if self.db_type == 'classification':
                if print_partial_progress is False:
                    if e == 0 or e == self.num_epochs - 1:
                        num_correct, total = self.zero_one_loss()
                        print('Results of epoch {}: {} / {} correct'.format(e + 1, num_correct, total))
                        self.start = num_correct
                else:
                    num_correct, total = self.zero_one_loss()
                    print('Results of epoch {}: {} / {} correct'.format(e + 1, num_correct, total))
                    self.end = num_correct
            elif self.db_type == 'regression':
                if e == 0 or e == self.num_epochs - 1:
                    average_diff = self.regression_difference()
                    print('Results of epoch {}: {}'.format(e, average_diff))

    '''
    ----------------------------------------------
    Basically finding the partial derivatives of
    the cost with respect to both the weights and the biases
    '''
    def back_prop(self, in_act_vec, desired_out):

        # Variable delcarations
        der_b = [np.zeros(b.shape) for b in self.bias_vec]
        der_w = [np.zeros(w.shape) for w in self.weight_vec]

        act = in_act_vec
        act_vec = [in_act_vec]
        z_vecs = []

        # For every weight vector and respective layer bias,
        # find every layer's pre-and-post-sigmoid activation vector
        for curr_b, curr_w in zip(self.bias_vec, self.weight_vec):

            # print('\n\ncurr_w')
            # print(curr_w)
            # print('\n\nact')
            # print(act)
            z = np.dot(curr_w, act) + curr_b
            z_vecs.append(z)
            act = (self.act_fn)(z)
            act_vec.append(act)

        # Notice this is the same as the "for layer_idx..." loop below.
        # We need to do this first step at the last layer in
        # a particular way, so it goes outside of the loop

        delta_l = self.cost_prime(act_vec[-1], desired_out) * sig_prime(z_vecs[-1])
        der_b[-1] = delta_l
        der_w[-1] = np.dot(delta_l, act_vec[-2].transpose())
        for L in range(2, len(self.layer_sizes)):

            z = z_vecs[-L]
            delta_l = np.dot(self.weight_vec[-L+1].transpose(), delta_l) * sig_prime(z)
            der_b[-L] = delta_l
            der_w[-L] = np.dot(delta_l, act_vec[-L-1].transpose())

        return (der_b, der_w)

    '''
    ----------------------------------------------
    The derivative of our cost function
    if the cost function is (a - y)^2

    TODO: see if not multiplying by 2 is fine too
    '''
    def cost_prime(self, out_acts, desired_out):
        return out_acts - desired_out
    
    '''
    ----------------------------------------------
    Returns the percentage of correct classifications
    '''
    def zero_one_loss(self):

        num_correct = 0
        total = len(self.old_data)

        for actual_out, desired_out in self.old_data:
            if np.argmax((self.feed_forward(actual_out))) == np.argmax(desired_out):
                num_correct += 1
        
        return (num_correct, total)
    
    def regression_difference(self):
        distances = []

        for actual_out, desired_out in self.old_data:
            distances.append(abs(self.feed_forward(actual_out) - desired_out))
        return np.mean(distances)

'''
----------------------------------------------
w_dot_a_plus_b   A weighted sum which equals
                 the dot product of the weight vector (w)
                 and activation vector (a)
                 plus the bias vector (b)

Vectorizes sigmoid over
Sigmoid in math notation: https://www.oreilly.com/library/view/hands-on-automated-machine/9781788629898/assets/1e81ffa6-1c5a-4d48-b88e-d01d4047f2ef.png
'''
def sig(w_dot_a_plus_b):
    return 1.0 / (1.0 + np.exp(-w_dot_a_plus_b))

'''
----------------------------------------------
The derivative of the sigmoid function
'''
def sig_prime(w_dot_a_plus_b):
    return sig(w_dot_a_plus_b) * (1 - sig(w_dot_a_plus_b))

def one_times(x):
    return x