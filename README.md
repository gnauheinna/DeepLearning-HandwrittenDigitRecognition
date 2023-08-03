
## DeepLearning-HandwrittenDigitRecognition


# This python project is a feedforward neural network that classifies handwritten digits. 
The project includes three neural network modules that is trained using the MNIST dataset.

The project include three modules, network.py, network2.py, and network3.py (with the latter modules as improved versions of the first network). 
1. Networt.py is a module that implements the stochastic gradient descent learning algorithm for a feedforward neural network.  Gradients are calculated using backpropagation. 
2.  Network2.py is an improved version of network.py, also implementing the stochastic gradient descent learning algorithm for a feedforward neural network. The improvements include the addition of the cross-entropy cost function, regularization, and better initialization of network weights.
3.  Network3.py is Theano-based program for training and running simple neural networks. It supports several layer types (fully connected, convolutional, max pooling, softmax), and activation functions (sigmoid, tanh, and rectified linear units, with more easily added). Network3 is much faster than the previous networks when run on a cpu.  However, unlike network.py and network2.py it can also be run on a GPU, which makes it faster still.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

