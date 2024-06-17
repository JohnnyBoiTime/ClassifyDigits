import numpy as np
import random

class Network(object):

    # constructor for the network
    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # A 2D array with y rows and 1 column, skips first layer ([1:])
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # sigmoid
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def derivSigmoid(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))
    
    def evaluate(self, testData):
        testResults = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)
    
    def derivCost(self, outputActivations, y):
        return (outputActivations - y)

    # feeds it forward
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    # 
    def updateMiniBatch(self, miniBatch, eta):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(x, y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)] 
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]
        self.weights = [w - (eta / len(miniBatch)) * nw
                        for w, nw in zip(self.weights, nablaW)]
        self.biases = [b - (eta / len(miniBatch)) * nb
                       for b, nb in zip(self.biases, nablaB)]

    def SGD(self, trainingData, epochs, miniBatchSize, eta, testData = None):
        if testData:
            nTest = len(testData)

        n = len(trainingData)

        for j in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [
                trainingData[k : k + miniBatchSize]
                for k in range(0, n, miniBatchSize)]
            
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)

            if testData:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), nTest))
            else:
                print("Epoch {0} complete".format(j))
            
    def backprop(self, x, y):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z) 
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.derivCost(activations[-1], y) * self.derivSigmoid(zs[-1])
        nablaB[-1] = delta 
        nablaW[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.numLayers):
            z = zs[-l]
            ds = self.derivSigmoid(z)
            delta=np.dot(self.weights[-l + 1].transpose(), delta) * ds
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nablaB, nablaW
    

