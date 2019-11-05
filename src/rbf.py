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
    def __init__(self, k, o, epochs=100):
        self.k = k
        self.epochs = epochs
        # Initialize matrix of weights
        self.weights = [np.random.rand(k) for i in range(o)]
        self.learn_rate = 0.1
        self.alpha = 0.5
    
    '''
    @brief              Fit the model
    @param X            The dataset
    @param y            The correct classifacation for each data example
    @param classes      The set of possible classes
    @param type         The dataset type (classification or regression)
    '''
    def fit(self, X, centers, y, type, classes=[]):
        # Compute standard deviations
        std_devs = get_std_devs(centers)

        for epoch in range(self.epochs):
            print("EPOCH: ", epoch)
            for i in range(len(X)):
                # Build array of gaussians for each center
                g = np.array([gaussian(X[i], c, s) for c,s in zip(centers, std_devs)])
                # Predict the class
                output_scores = [g.T.dot(self.weights[i]) for i in range(len(self.weights))]
                # print('OUTPUT SCORES: ', output_scores)
                F = max(output_scores) if type == 'regression' else output_scores.index(max(output_scores))
                # print("F = ", F)
                
                # Compute the loss (for classification, use 0-1)
                if type == 'classification':
                    loss = 0 if classes[F] == y[i] else 1
                else:
                    loss = (y[i] - F) ** 2
                
                error = 1 - F if type == 'classification' else y[i] - F
                
                # Update weights
                for w in self.weights:
                    w = w - 0.1 * g * 

    def predict(self, x, type):
        g = np.array([gaussian(x, c, s) for c,s in zip(centers, std_devs)])
        output_scores = [g.T.dot(self.weights[i]) for i in range(len(self.weights))]
        
        if type == "classification":
            F = output_scores.index(max(output_scores))
        else:
            F = max(output_scores)
        
        return F


                
                

                



    

                



