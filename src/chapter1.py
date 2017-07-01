import mnist_loader
import networkFast
import network2Fast

import numpy as np

np.random.seed(2**32-10)

trainingData, validationData, testData = mnist_loader.load_data_wrapper()

net = networkFast.Network([784, 30, 10])
net.sgd(trainingData, 30, 20, 3.0, validationData)


trainingData, validationData, testData = mnist_loader.load_data_wrapper()

net2 = network2Fast.Network([784, 30, 10], cost=network2Fast.CrossEntropyCost)
net2.large_weight_initializer()
net2.sgd(trainingData, 30, 20, 0.5, evaluation_data=validationData, monitor_evaluation_accuracy=True)


trainingData, validationData, testData = mnist_loader.load_data_wrapper()

net3 = network2Fast.Network([784, 30, 10], cost=network2Fast.CrossEntropyCost)
net3.large_weight_initializer()
net3.sgd(trainingData, 30, 20, 0.5, evaluation_data=validationData, monitor_evaluation_accuracy=True)

# netOne = network.Network([784, 10])
# netOne.sgd(trainingData, 30, 10, 3.0, testData)


# netDeeper = network.Network([784, 50, 50, 10])
# netDeeper.sgd(trainingData, 30, 10, 3.0, testData)
