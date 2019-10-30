'''
This is code I found at https://pythonmachinelearning.pro/using-neural-networks-for-regression-radial-basis-function-networks/
It is an example implementation of a radial basis function network that I am using to 
'''
import numpy as np

'''
@param s    the std deviation
@param c    the cluster center (or neuron)
'''
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

class RBFNet(object):
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k # This will be determined by ENN or CNN
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k) # Initialize weights to random values
        self.b = np.random.randn(1) # Initialize bias to be a random number

    '''
    @breif      Compute weights and biases
    @param X    The training set
    @param y    The correct classes
    '''
    def fit(self, X, y):
        # If using kmeans(CNN/ENN)
        if self.inferStds:
            pass # We would use ENN or CNN here to find centers and compute each center's std dev
        else:
            # use fixed standard deviation, we'll probably not use this case
            pass
        
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                # create array of rbfs for this point (X[i]), one for each neuron in our hidden layer
                a = np.array([self.rbf(X[i], c, s) for c, s in zip(self.centers, self.std)])
                F = a.T.dot(self.w) + self.b # Predict the regression value
                # Calculate the loss - difference between actual class and predicted, squared
                loss = (y[i] - F).flatten() ** 2

                # backward pass
                error = -(y[i] - F).flatten()
                
                # update weights and bias
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
    

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([ self.rbf(X[i], c, s) for c, s in zip(self.centers, self.stds) ])


NUM_SAMPLES = 100
X = np.random.uniform(0., 1., NUM_SAMPLES) # Set of random floats between 0 and 1
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES) # Add some thicc noise
y = np.sin(2 * np.pi) # Classify X values with sine function




    

