"""network3.py
~~~~~~~~~~~~~~

A Tensorflow-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Tensorflow, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

Written for Tensorflow 1.1.0.

"""

# Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import pooling

# Activation functions for neurons
from tensorflow.python.ops.math_ops import sigmoid


def linear(z): return tf.identity(z)


def relu(z): return tf.maximum(0.0, z)


# Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    return [training_data, validation_data, test_data]


# Main class used to construct and train networks
class Network(object):
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tf.placeholder(tf.float32, shape=(None, 784), name="X")
        self.y = tf.placeholder(tf.int64, shape=None, name="y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def sgd(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data) // mini_batch_size
        num_validation_batches = size(validation_data) // mini_batch_size
        num_test_batches = size(test_data) // mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = tf.reduce_sum([tf.nn.l2_loss(layer.w) for layer in self.layers])
        cost = self.layers[-1].cost(self) + lmbda * l2_norm_squared / num_training_batches
        grads = tf.gradients(cost, self.params)

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.

        updates = [tf.assign(param, param - eta * grad) for param, grad in zip(self.params, grads)]

        validate_mb_accuracy = self.layers[-1].accuracy(self.y)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index

                sess.run(updates, feed_dict={self.x: training_x[minibatch_index * self.mini_batch_size: (minibatch_index + 1) * self.mini_batch_size],
                                             self.y: training_y[minibatch_index * self.mini_batch_size: (minibatch_index + 1) * self.mini_batch_size]})

                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [sess.run(validate_mb_accuracy, feed_dict={
                            self.x: validation_x[j * self.mini_batch_size: (j + 1) * self.mini_batch_size],
                            self.y: validation_y[j * self.mini_batch_size: (j + 1) * self.mini_batch_size]})
                            for j in range(num_validation_batches)])

                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [sess.run(validate_mb_accuracy, feed_dict={
                                    self.x: test_x[j * self.mini_batch_size: (j + 1) * self.mini_batch_size],
                                    self.y: test_y[j * self.mini_batch_size: (j + 1) * self.mini_batch_size]})
                                 for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))

        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
           best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


# Define layer types
class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        self.w = tf.Variable(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape)),
            dtype=tf.float32)
        self.b = tf.Variable(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[3],))),
            dtype=tf.float32)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, self.image_shape)
        conv_out = tf.nn.conv2d(
            input=self.inpt, filter=self.w, strides=[1, 1, 1, 1], padding="VALID")
        pooled_out = tf.nn.max_pool(
            value=conv_out, ksize=self.poolsize, strides=self.poolsize, padding="VALID")
        self.output = self.activation_fn(
            pooled_out + self.b)
        self.output_dropout = self.output  # no dropout in the convolutional layers


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = tf.Variable(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_in), size=(n_in, n_out))),
            dtype=tf.float32)
        self.b = tf.Variable(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=1.0, size=(n_out,))),
            dtype=tf.float32)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.p_dropout) * tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.arg_max(self.output, dimension=1)
        self.inpt_dropout = dropout_layer(
            tf.reshape(inpt_dropout, (mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            tf.matmul(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        tf.reduce_mean(tf.cast(tf.equal(y, self.y_out), tf.float32))


class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = tf.Variable(
            np.zeros((n_in, n_out)), dtype=tf.float32,
            name='w')
        self.b = tf.Variable(
            np.zeros((n_out,)), dtype=tf.float32,
            name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = tf.nn.softmax((1 - self.p_dropout) * tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.arg_max(self.output, dimension=1)
        self.inpt_dropout = dropout_layer(
            tf.reshape(inpt_dropout, (mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = tf.matmul(self.inpt_dropout, self.w) + self.b

    def cost(self, net):
        """Return the log-likelihood cost."""
        return tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.one_hot(net.y, 10, 1.0, 0.0, axis=-1),
                        logits=self.output_dropout))

    def accuracy(self, y):
        """Return the accuracy for the mini-batch."""
        return tf.reduce_mean(tf.cast(tf.equal(y, self.y_out), tf.float32))


# Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    # return data[0].get_shape().as_list()[0]
    return data[0].shape[0]


def dropout_layer(layer, p_dropout):
    mask = tf.contrib.keras.backend.random_binomial(shape=layer.shape, p=1 - p_dropout)
    #mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    #return layer * T.cast(mask, theano.config.floatX)
    return layer * mask
