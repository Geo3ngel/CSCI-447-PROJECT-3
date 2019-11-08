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

def get_column(matrix, i):
    return [row[i] for row in matrix]

from collections import Counter
def find_mean_pt(centers):
    vars = [get_column(centers, i) for i in range(len(centers[0]))]
    mean_pt = []
    for v in vars:
        if type(v[0]) == str:
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
    dists = [cf.euc_dist(mean, c1)**2 for c1 in c]
    return np.sqrt(sum(dists) / len(c))


def cost_func(output_scores, F, y, type, classes=[]):
    if type == 'classification':
        correct_probs = np.array([1 if c == y else 0 for c in classes])
        errors = (correct_probs - output_scores) ** 2
        return sum(errors)
    else:
        return (float(y) - float(F)) ** 2



'''
@param k        the number of centers/hidden layer nodes
@param o        the number of output nodes
@param epochs   the number of epochs to run for
'''
class RBF():
    def __init__(self, k, o, output_file, epochs=100):
        self.k = k
        self.epochs = epochs
        # Initialize matrix of weights
        self.weights = np.array([np.random.uniform(-0.1,0.1, k) for i in range(o)])
        # Initialize matrix of momentums
        self.momentums = np.array([np.ones(k) for i in range(o)])
        self.learn_rate = 0.001
        self.alpha = 0.005
        self.output_file = output_file
    
    '''
    @brief  update the momentum matrix for next iteration
    '''
    def update_momentum(self, errors, output_scores, activations):
        for i in range(len(self.momentums)):
            self.momentums[i] = self.learn_rate * -1 * ((errors[i] - output_scores[i]) * output_scores[i] * (1 - errors[i])) + (self.alpha * self.momentums[i])


    '''
    @param activations      the array of activations for this datapoint
    @param output_scores    the array of output scores
    @param F                the predicted class/regression value
    @param type             the type of dataset (classification/regression)
    @param y                the correct class/regression value
    @brief                  perform backprop
    '''
    def back_prop(self, activations, output_scores, F, type_, y, classes=[]):
        if type_ == "classification":
            correct_probs = np.array([1 if c == y else 0 for c in classes])
            errors = correct_probs - output_scores
            self.update_momentum(errors, output_scores, activations)
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - self.momentums[i]

        else: # Type is regression
            error = float(y) - float(F)
            self.update_momentum(np.array([error]), output_scores, activations)
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - self.momentums[i]       
    
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
        for epoch in range(self.epochs):
            print("EPOCH: ", epoch)
            self.output_file.write("EPOCH: " + str(epoch) + "\n")
            for i in range(len(X)):
                # Build array of gaussians for each center
                g = np.array([gaussian(X[i], c, s) for c in centers])
                # Compute scores for each output node
                output_scores = np.array([g.T.dot(self.weights[i]) for i in range(len(self.weights))])
                # Guess the class/regression value
                F = max(output_scores) if type == 'regression' else np.where(output_scores == np.max(output_scores))[0][0]
                # Then do back prop
                self.back_prop(g, output_scores, F, type, y[i], classes)
                cost = cost_func(output_scores, F, y[i], type, classes)
                self.output_file.write("COST: " + str(cost) + "\n")
                if np.isnan(cost):
                    break
                # print("COST: ", cost)
            
            self.test(X,type, y, centers, classes)
            print('\n\n')

            
    def test(self, X, type, y, centers, classes=[]):
        print("TESTING:")
        if type == 'classification':
            correct_guesses = 0
            for i in range(len(X)):
                guess = self.predict(X[i], type, centers)
                if type == 'classification':
                    if classes[guess] == y[i]:
                        correct_guesses += 1
            print("GUESSES: ", correct_guesses, "/", len(X))
            self.output_file.write("GUESSES: " + str(correct_guesses) + "/" + str(len(X)) + "\n")
        else: # regression
            avg_guess = 0
            for i in range(len(X)):
                guess = self.predict(X[i], type, centers)
                avg_guess += np.abs(float(guess) - float(y[i]))
            print("AVG DIFF", avg_guess / len(X))
            self.output_file.write("AVG DIFF " + str(avg_guess / len(X)) + "\n")
            

    def predict(self, x, type, centers):
        s = std_dev(centers)
        g = np.array([gaussian(x, c, s) for c in centers])
        output_scores = [g.T.dot(self.weights[i]) for i in range(len(self.weights))]
        # print(output_scores)
        if type == "classification":
            F = output_scores.index(max(output_scores))
        else:
            F = max(output_scores)
        return F


                
                

                



    

                



