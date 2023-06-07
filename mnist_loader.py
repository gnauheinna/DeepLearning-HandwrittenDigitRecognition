"""
mnist_loader
@gnauheinna
A library to load the MNIST image data
"""

# Libraries
import pickle as cp
import gzip
import numpy as np

def load_data():
    """ 
    Return the MNIST data as a tuple containing the training dat, the validation data, and the test data
    1. The "training_data" os returned as a tuple with two entries. I. The first entry contains the actual training images 
    (a numpy ndarray with 50,000 entries, each entries with 784 values, representing 28 * 28 pixels in a single MNIST image)
    II. The second entry is a numpy ndarray containing 50,000 entries that are digit values (0-9) for the corresponding images contained int he first entry of the tuple.
    2. The "validation_data" and "test_data" are similar, each containing 10,000 images 
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cp.load(f)
    f.close()
    return(training_data, validation_data, test_data)

def load_data_wrapper():
    """
        returns a tuple containing "(training_data, validation_data, test_data)"
        but the format is more convenient for use in this implementation of neural networks
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] 
    training_results = [vectorized_result(y) for y in tr_d[1]] 
    training_data = zip(training_inputs , training_results) 
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]] 
    validation_data = zip(validation_inputs , va_d[1]) 
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]] 
    test_data = zip(test_inputs , te_d[1]) 
    return (training_data , validation_data , test_data)

def vectorized_result(j):
    """
        returns a 10-dimensional unit vector with a 1.0 in the jth position and zero
        elsewher. This is used to convert a digit (0-9) into a corresponding desired 
        output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

