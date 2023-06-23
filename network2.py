"""
    network2.py - QuadraticCost, CrossEntropyCost, and Network Class
    @gnauheinna
"""

# Libraries
import numpy as np


##### Quadratic Cost



##### Cross-entropy cost functions

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """ a measure of how well an output activation, a, matches its desired output, y.
            C = 1/n * ∑(Xj (σ(z)-7))
        """

        # np.nan_to_num ensures that the np deals correctly with the log of numbers very close to zero
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a , y):
        """ tells our network how to compute the output error. σ^l = a^l - y
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

