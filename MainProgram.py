from NeuralNetwork import Network
import tensorflow as tf
import numpy as np

# Load the data to be used
(trainingImages, trainingLabels) , (testImages, testLabels ) = tf.keras.datasets.mnist.load_data()

# Changes pixel values to between 0 and 1
trainingImages = trainingImages / 255.0
testImages = testImages / 255.0

# 784
trainingImages = trainingImages.reshape((trainingImages.shape[0], 784, 1))
testImages = testImages.reshape((testImages.shape[0], 784, 1))

# converting the labels to one-hot vectors (0-9)
def oneHot(labels, numClasses = 10):
    return np.eye(numClasses)[labels].reshape(-1, numClasses, 1)

# Converting labels to readable
trainingLabels = oneHot(trainingLabels)
testLabels = oneHot(testLabels)

# Pairs data
trainingData = list(zip(trainingImages, trainingLabels))
testData = list(zip(testImages, testLabels))

# Network
net = Network([784, 30, 10])
net.SGD(trainingData, epochs=30, miniBatchSize=10, eta=3.0, testData=testData)

net.visualize(testData)

