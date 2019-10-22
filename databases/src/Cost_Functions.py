# This class contains static functions for cost functions

# IMPORTS
import numpy as np

## Quadratic Costs: returns the cost associated with the given output & desired output
@staticmethod
def get_quadratic_cost(output, desired_output):
    return 0.5*np.linalg.norm(output - desired_output)**2

# Returns the quadratic error
@staticmethod
def get_quad_error_delta(z_val, output, desired_output):
    # Our z value is the resulting summation of our nodes weights, activation values, and bias.
    return (output-desired_output) * (calc_sigmoid(z_val)*(1-calc_sigmoid(z_val)))

## Cross Entropy Cross:
def get_cross_entropy_cost(output, desired_output):
    # Do delta in these
    return np.sum(np.nan_to_num(-desired_output*np.log(output)-(1-desired_output)*np.log(1-output)))

# Returns the cross entropy error
def get_cross_entroy_error_delta(output, desired_output):
    return output - desired_output
    

## Mean Squared Error
def get_MSE():
    # TODO: write me
    pass

## General Utility

# Generates a sigmoid value given the 
def calc_sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# (1.0/(1.0+np.exp(-z)))*(1-sigmoid(z))