import mnist_loader
import networkFast
import network

import numpy as np

np.random.seed(2**32-10)

trainingData, validationData, testData = mnist_loader.load_data_wrapper()

#net = networkFast.Network([784, 30, 10])
net = network.Network([784, 30, 10])
net.SGD(trainingData, 30, 20, 3.0, testData)


# netOne = network.Network([784, 10])
# netOne.SGD(trainingData, 30, 10, 3.0, testData)


# netDeeper = network.Network([784, 50, 50, 10])
# netDeeper.SGD(trainingData, 30, 10, 3.0, testData)
