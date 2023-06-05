import numpy as np
import random
# The Network Class is the center piece of the project; it represents the neutral network 

class Network(object):

    def __init_(self, sizes):
    # a network object takes in the parameter size, which is a list that contains 
    # the number of neurons in the repective  layers.
    # There are four fields of a Network object: 1) num_layers represents the number of layers of the neutral network,
    # 2) sizes is the parameter, 3) biases and 4) weights are initialized randomly to generate Gaussian distribution
    # with the mean zero and standard deviation 1.

        self.num_layers = len(sizes)
        self.sizes = sizes

        # the network initialization code assumes that the first layer of neurons as the input layer 
        # and the last layer as the output layer. It omits to set bias and weights for those layers.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes [1:])]

    
    # a' = O(wa+b) : how to obtain the activation of the ith layer from the i-1th layer
    # sigmoid function: takes an input z as a vector/ Numpy array
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    # Feedforward method: given an input a for the network returns the corresponding output. the method applies a'  
    def feedforward(self, a):
        """ Return the output of the network if "a" is input"""
        for b, w  in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
        return a
    
    # SGD method that implements stochastic gradient descent
    # In each epoch, the code randomly shuffles the training data and then partitions it into mini-batches of the appropriate size
    # Then for each mini_batch we apply a single step of gradient descetn, using just the training data of mini_batch

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Training the neural network using mini-batch stochastic gradient descent. The 
            "training_data" is a list of tuples "(x, y)" representing the training inputs and the desired outputs.
            "epochs" represents the number of epochs to train for
            "mini_batch_size" represents the size of the mini-batches to use when sampling
            "eta" is the learning rate
            If "test_data" is provided then the network will be evaluated against the test data after each epoch 
            and partial progess printed out. This is useful for tracking progress, but slows things down substantially.
        """
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_minibatch(mini_batch, eta)
        if test_data:
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
        else:
            print ("Epoch {0} complete".format(j))

    def update_minibatch(self, mini_batch, eta):
        """ Updates the network's weights and biases by applying gradient descent using 
            backpropagation to a single mini batch.
            "mini_batch" is a list of tuples(x, y), and "eta" is the learning rate.
        """
        nabla_b =[np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # invokes backpropagation algorithm (computes the gradient: sigmoid_prime, which computes the dericative of the
            # sigmiod function and self.cost_detivative
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]