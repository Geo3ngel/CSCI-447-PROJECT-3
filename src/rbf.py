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
@brief      return a set of standard deviations for each center/neuron
'''
def get_std_devs(c):
    return [sum([cf.euc_dist(c1,c2) for c2 in c]) / len(c) for c1 in c]

from collections import Counter
def find_mean_pt(centers):
    vars = [np.array(centers)[:,i].tolist() for i in range(len(centers[0]))]
    mean_pt = []
    for v in vars:
        print("V: ", v)
        # if v.dtype.type is np.str_:
        if type(v[0]) == str:
            print("It's a string.")
            c = Counter(v)
            mean_pt.append(c.most_common(1)[0][0])
        else:
            mean_pt.append(np.mean(np.array(v)))
    return mean_pt
    
        
            

'''
@param  c the centers
@brief  get the std deviation for the entire set of centers 
'''
def std_dev(c):
    # d_max = max([cf.euc_dist(c1,c2) for c1 in c for c2 in c])
    # return d_max / np.sqrt(2 * len(c))
    mean = find_mean_pt(c)
    print("MEAN POINT:")
    print(mean)
    dists = [cf.euc_dist(mean, c1) for c1 in c]
    return sum(dists) / len(c)

def cost_func(output_scores, F, y, type, classes=[]):
    if type == 'classification':
        correct_probs = np.array([1 if c == y else 0 for c in classes])
        errors = (correct_probs - output_scores) ** 2
        return sum(errors)
    else:
        return (y - F) ** 2



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
        self.weights = np.array([np.zeros(k) for i in range(o)])
        self.learn_rate = 0.1
        self.alpha = 0.5
    
    '''
    @param activations      the array of activations for this datapoint
    @param output_scores    the array of output scores
    @param F                the predicted class/regression value
    @param type             the type of dataset (classification/regression)
    @param y                the correct class/regression value
    @brief                  perform backprop
    '''
    def back_prop(self, activations, output_scores, F, type, y, classes=[]):
        if type == "classification":
            correct_probs = np.array([1 if c == y else 0 for c in classes])
            errors = correct_probs - output_scores
            # print("ERRORS: ", errors)
            # print("Output Scores: ", output_scores)
            
            # print("WEIGHTS: ")
            # print(self.weights)
            
            # print("Errors: ", errors)
            # for w,a in zip(self.weights.T, activations):
            for i in range(len(self.weights.T)):
                print("BEFORE WT: ")
                print(self.weights.T)
                self.weights.T[i] = self.weights.T[i] - self.learn_rate * errors * self.alpha * activations[i]
                print("AFTER WT: ") 
                print(self.weights.T)
            
            print("UPDATED WEIGHTS:") 
            print(self.weights)

        else: # Type = regression
            error = y - F
            for w in self.weights:
                w = w - self.learn_rate * error * activations           
    
    '''
    @brief              Fit the model
    @param X            The dataset
    @param y            The correct classifacation for each data example
    @param classes      The set of possible classes
    @param type         The dataset type (classification or regression)
    '''
    def fit(self, X, centers, y, type, classes=[]):
        # Compute standard deviations
        s = std_dev(centers)
        print("STD DEV: ", s)
        
        for epoch in range(self.epochs):
            print("EPOCH: ", epoch)
            for i in range(len(X)):
                print("POINT ", X[i])
                # Build array of gaussians for each center
                g = np.array([gaussian(X[i], c, s) for c in centers])
                # print("GAUSSIANS:")
                # print(g)
                # Compute scores for each output node
                output_scores = np.array([g.T.dot(self.weights[i]) for i in range(len(self.weights))])
                print("OUTPUT SCORES: ")
                print(output_scores)
                # Guess the class/regression value
                F = max(output_scores) if type == 'regression' else np.argmax(output_scores)
                print("F: ", F)
                # Then do back prop
                self.back_prop(g, output_scores, F, type, y[i], classes)
                # print("WEIGHTS: ")
                # print(self.weights)
                cost = cost_func(output_scores, F, y[i], type, classes)
                print("COST: ", cost)
                
            print('\n\n')

            
            

    def predict(self, x, type, centers):
        std_devs = get_std_devs(centers)
        g = np.array([gaussian(x, c, s) for c,s in zip(centers, std_devs)])
        output_scores = [g.T.dot(self.weights[i]) for i in range(len(self.weights))]
        print("OUTPUT SCORES: ", output_scores)
        if type == "classification":
            F = output_scores.index(max(output_scores))
        else:
            F = max(output_scores)
        return F


                
                

                



    

                



