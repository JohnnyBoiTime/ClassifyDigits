import numpy as np
import matplotlib.pyplot as plt
import random

# CREDIT TO MICHAEL NIELSEN FOR CODE

# Cross Entropy 
class CrossEntropyCost(object):

    # Cost function
    @staticmethod
    def equation(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y)*np.log(1 - a)))

    # Compare activation a with desired output y
    @staticmethod
    def compare(z, a, y):
        return (a - y)
    
# Quadratic Cost    
class QuadraticCost(object):

    # Quadratic Cost Function
    @staticmethod
    def equation(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    # Compare activation a with desired output y
    @staticmethod
    def compare(z, a, y):
        return (a - y) * Network.derivSigmoid(z) 

class Network(object):

    # constructor for the network
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.numLayers = len(sizes)
        self.sizes = sizes
        # self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # A 2D array with y rows and 1 column, skips first layer ([1:])
        # self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.defaultWeightInitializer()
        self.cost = cost

    # Initialize weights (with squishing)
    def defaultWeightInitializer(self):
        self.bias = [np.random.rand(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.rand(y,x) / np.sqrt(x) # Squish weight values to middle
                        for x, y in zip(self.sizes[:-1], self.sizes)]

    # Initialize weights (without squishing)
    def largeWeightInitializer(self):
        self.bias = [np.random.rand(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.rand(y, x)
                        for x, y in zip(self.sizes[:1], self.sizes[1:])]

    # sigmoid
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # sigmoid derivative
    @staticmethod
    def derivSigmoid(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))
    
    # Checks accuracy
    def evaluate(self, testData):
        testResults = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)
    
    # Change in cost of network
    def derivCost(self, outputActivations, y):
        return (outputActivations - y)

    # feeds it forward
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    # Adjusts weights and biases in mini batch
    def updateMiniBatch(self, miniBatch, eta, lmbda, n):

        # Store gradients
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        # Go throgh minibatches
        for x, y in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(x, y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)] 
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]

        # Update weight with regularization
        self.weights = [(1 -  eta * (lmbda / n)) * w - (eta / len(miniBatch)) * nw 
                        for w, nw in zip(self.weights, nablaW)]
        self.biases = [b - (eta / len(miniBatch)) * nb
                       for b, nb in zip(self.biases, nablaB)]

    # Stochastic gradient descent
    def SGD(self, trainingData, epochs, miniBatchSize, eta, 
            lmbda = 0.0,
            evaluationData = None,
            monitorEvaluationCost=False,
            monitorEvaluationAccuracy=False,
            monitorTrainingCost=False,
            monitorTrainingAccuracy=False):
        
        # Evaluation data
        if evaluationData:
            nData = len(evaluationData)

        # Samples
        n = len(trainingData)

        # Store evaluation and training
        evaluationCost, evaluationAccuracy = [], [] 
        trainingCost, trainingAccuracy = [], []

        for j in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [
                trainingData[k : k + miniBatchSize]
                for k in range(0, n, miniBatchSize)]
            
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta, lmbda, len(trainingData))
            
            print(f"Epoch {j} training complete")

            if monitorTrainingCost:
                cost = self.totalCost(trainingData, lmbda)
                trainingCost.append(cost)
                print(f"Cost on training data: {cost}")
            
            if monitorTrainingAccuracy:
                accuracy = self.accuracy(trainingData, convert=True)
                trainingAccuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy}, {n}")

            if monitorEvaluationCost:
                cost = self.totalCost(evaluationData, lmbda, convert=True)
                evaluationCost.append(cost)
                print(f"Cost on evaluation data: {cost}")

            if monitorEvaluationAccuracy:
                accuracy = self.accuracy(evaluationData)
                evaluationAccuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(evaluationData)} {nData}")

        return evaluationCost, evaluationAccuracy, trainingCost, trainingAccuracy


            
    # Backpropagation        
    def backprop(self, x, y):

        # Store gradients
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        activation = x # Initialize input activation
        activations = [x] # stores all activations starting with input layer
        zs = [] # stores all weighted input vectors z

        
        # Feedfoward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z) 
            activation = self.sigmoid(z)
            activations.append(activation)

        # [-1] is accessed the last element of the list
        delta = self.derivCost(activations[-1], y) * self.derivSigmoid(zs[-1])
        nablaB[-1] = delta 
        nablaW[-1] = np.dot(delta, activations[-2].transpose())

        # moves backwards for backprop
        for l in range(2, self.numLayers):
            z = zs[-l]
            ds = self.derivSigmoid(z)
            delta=np.dot(self.weights[-l + 1].transpose(), delta) * ds
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nablaB, nablaW)
    
    # Compares prediction to true label
    def accuracy(self, data, convert=False):
        """
         Since feedforward produces a vector of probabilities, the index with
         the highest probability is what is returned. That index is compared
         with the one hot vector true label. 
         
         Example:
          feedfoward = [0.78, 0.23, 0.15] - indicates prediction is a 0
          
          one hot (y) = [1, 0, 0] - indicates true label is a 0
          
          They are the same! Thus the prediction is correct. 

          Returns number of correct predictions.

          """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x,y) in data]
            
        else:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]

        # Sum number of correct predictions    
        return sum(int( x == y) for (x, y) in results)
    
    def totalCost(self, data, lmbda, convert=False):
        """
        
        
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = self.vectorizedResult(y)

    # Returns a one-hot vector, having a 1 where the result is
    def vectorizedResult(j):
        result = np.zeros((10, 1)) # 10 rows 1 column 
        result[j] = 1.0
        return result
