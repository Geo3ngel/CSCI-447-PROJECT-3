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



'''
@param k        the number of centers/hidden layer nodes
@param o        the number of output nodes
@param epochs   the number of epochs to run for
'''
class RBF():
    def __init__(self, k, o, epochs):
        self.k = k
        self.epochs = epochs
        # Initialize matrix of weights
        self.weights = [np.random.rand(k) for i in range(o)]
    
    '''
    @brief              Fit the model
    @param X            The dataset
    @param y            The correct classifacation for each data example
    @param classes      The set of possible classes
    '''
    def fit(self, X, centers, y, classes=[]):
        # Compute standard deviations
        std_devs = get_std_devs(centers)

        for epoch in range(self.epochs):
            for row in X:
                print("ROW: ", row)
                # Build array of gaussians for each center
                g = np.array([gaussian(row, c, s) for c,s in zip(centers, std_devs)])
                # Predict the class
                output_scores = [g.T.dot(self.weights[i]) for i in range(len(self.weights))]
                # print('OUTPUT SCORES: ', output_scores)
                F = max(output_scores)
                # print("F: ", F)
                # Compute the loss (for classification, use 0-1?)
                C = 0 if F == y[i] else 1
                



    

                



