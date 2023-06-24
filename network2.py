"""
    network2.py - QuadraticCost, CrossEntropyCost, and Network Class
    @gnauheinna
"""

# Libraries
import numpy as np


##### Quadratic Cost



##### Cross-entropy cost functions

class CrossEntropyCost(object):
    """ The Cross Entropy Cost Function is an improvement from the quadratic cost function in network.py. Unlike the quadratic cost function, 
        the Cross-Entropy Cost Function avoids the problem of the learning slowing down because if the neuron's actual output is close to the desired
        output for all training inputs, z, then the cross-entropy will be close to zero. The cross-entropy learns faster if the initial error is larger. 
    """
    @staticmethod
    def fn(a, y):
        """ a measure of how well an output activation, a, matches its desired output, y.
            C = 1/n * ∑(Xj (σ(z)-7))
            Return the cost associated with an output "a" and desired output "y" Np.nan_to_num ensure numerical stability. If both "a" and "y"
            have a 1.0 in the same slot then the expresion (1-y)*np.log(1-a) returns nan. the np.nan_to_num rensures that this converted to the correct 
            value of (0.0).
        """
        # np.nan_to_num ensures that the np deals correctly with the log of numbers very close to zero
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a , y):
        """ tells our network how to compute the output error. σ^l = a^l - y
            Returns the erro delta from the output layer. Although the parametr z is not used in the function it is included in the method's 
            parameters in order to make the interface consistent with the delta method for other cost classes.
        """
        return (a-y)

##### Main Network Clas
class Network(object):
    """
        defaults cost to cross-entropy
    """
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weigths = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weigths = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

