import mnist_loader
import networkFast
import network2Fast
import network4

import numpy as np

T = network4.load_data_shared()

np.random.seed(2**32-10)

trainingData, validationData, testData = mnist_loader.load_data_wrapper()

net = networkFast.Network([784, 30, 10])
net.sgd(trainingData, 30, 20, 3.0, validationData)


net2 = network2Fast.Network([784, 30, 10], cost=network2Fast.CrossEntropyCost)
net2.large_weight_initializer()
net2.sgd(trainingData, 30, 20, 0.5,
         evaluation_data=validationData,
         monitor_evaluation_accuracy=True)


net3 = network2Fast.Network([784, 30, 10], cost=network2Fast.CrossEntropyCost)
net3.sgd(trainingData, 30, 20, 0.5,
         evaluation_data=validationData,
         monitor_evaluation_accuracy=True,
         monitor_training_accuracy=True)


net4 = network2Fast.Network([784, 30, 10], cost=network2Fast.CrossEntropyCost)
net4.sgd(trainingData, 30, 20, 0.5,
         lmbda=5.0,
         evaluation_data=validationData,
         monitor_evaluation_accuracy=True,
         monitor_training_accuracy=True)


net5 = network2Fast.Network([784, 100, 10], cost=network2Fast.CrossEntropyCost)
net5.sgd(trainingData, 30, 20, 0.5,
         lmbda=5.0,
         evaluation_data=validationData,
         monitor_evaluation_accuracy=True,
         monitor_training_accuracy=True)
