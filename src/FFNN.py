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
    def __init__(self, layer_sizes, db_type,
        data, desired_out_col, learning_rate,
        class_list=None, num_epochs=50, using_test_data=True, debug=True):

        self.debug = debug

        if self.debug is True:
            print('FFNN: Enter __init__')

        # A list of integers representing
        # the number of nodes in layer i
        self.layer_sizes = layer_sizes
        self.epochs = [[] for x in range(num_epochs)]
        self.db_type = db_type
        self.learning_rate = learning_rate
        self.data = self.split_and_augment_data(data, desired_out_col)
        if class_list:
            self.class_list = class_list


        # Initializes weights via a normal distribution.
        self.weight_vec = [np.random.randn(x, y)
            for x, y in zip(layer_sizes[1:], layer_sizes[:-1])]

        # Initializes biases via a normal distribution.
        #
        # We don't put a bias on the weights between
        # the input layer and the first layer,
        # hence layer_sizes[1:].
        self.bias_vec = [np.random.randn(x, 1)
            for x in self.layer_sizes[:len(layer_sizes) - 1]]

        self.grad_desc()

        print('\n\nEND WOOOOOO')
        print('\nself.weight_vec')
        print(self.weight_vec)
        print('\nself.bias_vec')
        print(self.bias_vec)

    '''
    ----------------------------------------------
    The counterpart to augment_data().
    This function is for when we want to test the neural net using
    test / training data. (To do this should be much slower.)
    '''
    def split_and_augment_data(self, data, desired_out_col):

        if self.debug is True:
            print('FFNN: Enter split_and_augment_data()')

        random.shuffle(data)

        temp = []
        for ex in data:
            desired = ex.pop(desired_out_col)
            new_ex = [attr for idx, attr in enumerate(ex)
                if idx != desired_out_col]
            temp.append((new_ex, desired))

        return temp

    # '''
    # ----------------------------------------------
    # The counterpart to split_and_augment_data().
    # This function is for when we want to only train the neural network,
    # which should be faster than if we were to also test the
    # trained neural network.
    # '''

    # def augment_data(self, data, desired_out_col):

    #     if self.debug is True:
    #         print('FFNN: Enter augment_data()')

    #     return [(ex, ex[desired_out_col]) for ex in data]

    # ----------------------------------------------
    # Returns the output layer produced from `in_act`
    #
    # in_act    An activation vector of some layer
    def feed_forward(self, in_act_vec):

        if self.debug is True:
            print('FFNN: Enter feed_forward()')

        for bias, weight in zip(self.weight_vec, self.bias_vec):
            out_act_vec = sig(np.dot(in_act_vec, weight) + bias)

        return out_act_vec

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
    def grad_desc(self, print_partial_progress=True):

        if self.debug is True:
            print('FFNN: Enter grad_desc()')

        # Variables to make the code cleaner
        # if self.test_data:
        #     n_test = len(self.test_data)

        n_data = len(self.data)
        n_epoch = len(self.epochs)
        len_batch = math.ceil(n_data / 10)

        # The gradient descent itself for every epoch
        for e in range(n_epoch):

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
                new_bias = [np.zeros(b.shape)
                    for b in self.bias_vec]
                new_weight = [np.zeros(w.shape)
                    for w in self.weight_vec]

                for ex, desired_out in curr_batch:
                    change_bias, change_weight = self.back_prop([sig(attr) for attr in ex], desired_out)
                    new_bias = [bias + change
                        for bias, change
                        in zip(new_bias, change_bias)]
                    new_weight = [weight + change
                        for weight, change
                        in zip(new_weight, change_weight)]

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
            # if print_partial_progress is False:
            #     if e == 0 or e == n_epoch - 1:
            #         num_correct, total = self.zero_one_loss()
            #         print('Epoch {}: {} / {}'.format(e, num_correct, total))
            # else:
            #     num_correct, total = self.zero_one_loss()
            #     print('Epoch {}: {} / {}'.format(e, num_correct, total))


    '''
    ----------------------------------------------
    Basically finding the partial derivatives of
    the cost with respect to both the weights and the biases
    '''
    def back_prop(self, in_act_vec, desired_out):

        # if self.debug is True:
        #     print('FFNN: Enter back_prop()')

        # Variable delcarations
        der_b = [np.zeros(b.shape)
            for b in self.bias_vec]
        der_w = [np.zeros(w.shape)
            for w in self.weight_vec]

        # A list of activation vectors per layer
        act_vec = [in_act_vec]

        # A list of weighted sums per layer
        # (i.e. activations before sigmoid)
        z = []

        # For every weight vector and respective layer bias,
        # find every layer's pre-and-post-sigmoid activation vector
        for curr_b, curr_w in zip(self.bias_vec, self.weight_vec):

            curr_a = np.dot(curr_w, in_act_vec) + curr_b
            # curr_act = np.dot(curr_weight, np.asarray(in_act_vec, dtype=curr_weight.dtype) + curr_bias
            
            z.append(curr_a)
            act_vec.append(sig(curr_a))

        # Notice this is the same as the "for layer_idx..." loop below.
        # We need to do this first step at the last layer in
        # a particular way, so it goes outside of the loop

        delta_l = self.cost_prime(act_vec[-1], desired_out) * sig_prime(z[-1])
        # delta_l = [np.sum(a) for a in delta_l]

        der_b[-1] = delta_l

        # We transpose because we need the list to be a column vector
        # change_weight[-1] = np.dot(temp, act_list[-2])

        # print('\n\nSTART DEBUG')
        # print('\ndelta_l')
        # print(delta_l)
        # print('\nact_vec[-2].transpose()')
        # print(act_vec[-2].transpose())

        der_w[-1] = np.dot(np.column_stack(delta_l), act_vec[-2].transpose())

        # print('\nder_w[-1]')
        # print(der_w[-1])

        for L in range(2, len(self.layer_sizes)):

            # Start at the last layer
            curr_sum = z[-L]

            # print('\n\nSTART DEBUG')
            # print('\nself.weight_vec[-L+1].transpose()')
            # print(self.weight_vec[-L+1].transpose())
            # print('\ndelta_l')
            # print(delta_l)
            # print('\nsig_prime(z[-L])')
            # print(sig_prime(z[-L]))
            # print('\ncolumn_stack(self.weight_vec[-L+1].transpose())')
            # print(np.column_stack(self.weight_vec[-L+1].transpose()))

            delta_l = np.dot(self.weight_vec[-L+1].transpose(), np.column_stack(delta_l)) * sig_prime(z[-L])
            # delta_l = [np.sum(a) for a in delta_l]

            # print('\nact_vec')
            # print(act_vec[-L-1])

            der_b[-L] = delta_l
            der_w[-L] = np.dot(delta_l, np.array(act_vec[-L-1]).transpose())
        return (der_b, der_w)

    '''
    ----------------------------------------------
    The derivative of our cost function
    if the cost function is (a - y)^2

    TODO: see if not multiplying by 2 is fine too
    '''
    def cost_prime(self, out_acts, desired_out):
        return 2*(out_acts-desired_out)
    
    '''
    ----------------------------------------------
    Returns the percentage of correct classifications
    '''
    def zero_one_loss(self):

        num_correct = 0
        total = len(self.test_data)

        if self.debug is True:
            correctly_classified = []

        for actual_out, desired_out in self.test_data:
            if np.argmax(self.feed_forward(actual_out)) == desired_out:
                num_correct += 1
                if self.debug is True:
                    correctly_classified.append((actual_out, desired_out))
        
        return (num_correct, total)
        

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