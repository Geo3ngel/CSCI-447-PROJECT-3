import numpy as np


def gaussian(x,c,s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)


class RBF():
    def __init__(self, k, epochs):
        self.k = k
        self.epochs = epochs
        # Initialize matrix of weights
        self.weights = [np.random.rand(k) for i in range(k)]
    
    '''
    
    @brief  Fit our model
    '''
    def predict(self, X, cluster_func):
        # Get cluster centers


        for epoch in range(self.epochs):
            for row in X:
                # Build array of gaussians for each center
                g = [gaussian(row, c, s) for c in self.centers]

