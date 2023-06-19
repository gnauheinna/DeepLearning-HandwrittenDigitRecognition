"""
network.py
@gnauheinna
The Network Class is the center piece of the project. It is a module to implement the stochastic descent learning
algorithm for a feedforward neural network. Gradients are calculated using backpropagation. 
"""

# Libraies
import numpy as np
import random

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

    def backprop(self, x, y):
        """
            Returns a tuple (nabla_b, nabla_w) representing the gradient for the cost function 
            ("nabla_b" and "nabla_w" are layer by layer lists of numpy arrays, similar to "self.biases" and "self.weights")
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]       # list to store all the activations layer by layer
        zs = []                 # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivatve(activations[-1], y) *self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # the variable 1 in the loop below is used this way: 
        # 1= 1 means the last layer of neurons, 1 = 2 is the second last layer of neurons and so -on
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        """
            REturn the vector of partial derivatives \partial{} C_x /\ partial{} a
            for the output activations
        """
        return(output_activations-y)
    
    def sigmoid_prime(self,z):
        """Derivative of the sigmoid function"""
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
