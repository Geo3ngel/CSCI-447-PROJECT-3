import numpy as np
import Cost_Functions as cf

'''
@param x    data example
@param c    cluster center
@param s    standard deviation
'''
def gaussian(x, c, s):
    return np.exp((-1 / (2*(s**2))) * (cf.euc_dist(x,c)**2))

'''
@param c    the set of centers
@brief      return a set of std deviations for each center/neuron
'''
def get_std_devs(c):
    return [sum([cf.euc_dist(c1,c2) for c2 in c]) / len(c) for c1 in c]



class RBF():
    def __init__(self, k, epochs):
        self.k = k
        self.epochs = epochs
        # Initialize matrix of weights
        self.weights = [np.random.rand(k) for i in range(k)]
    
    '''
    @brief      Fit the model
    @param X    The dataset
    '''
    def predict(self, X, centers):
        # Compute standard deviations
        std_devs = get_std_devs(centers)

        for epoch in range(self.epochs):
            for row in X:
                # Build array of gaussians for each center
                g = [gaussian(row, c, s) for c,s in zip(centers, std_devs)]
                print("GAUSSIANS:")
                print(g)



