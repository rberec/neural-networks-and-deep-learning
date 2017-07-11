import numpy as np
from matplotlib import pyplot as plt


def plot_overlay(test_accuracy,
                 training_accuracy,
                 num_epochs,
                 xmin,
                 training_set_size):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs),
            [accuracy/100.0 for accuracy in test_accuracy],
            color='#2A6EA6',
            label="Accuracy on the test data")
    ax.plot(np.arange(xmin, num_epochs),
            [accuracy*100.0/training_set_size
             for accuracy in training_accuracy],
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([90, 100])
    plt.legend(loc="lower right")
    plt.show()