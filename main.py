"""
  Testing code for different network configuration
  
  Usage in shell:
        python3 main.py
  
  Parameters: (for Network.py and Network2.py)
      2nd parameter: epochs count
      3rd parameter: batch size
      4th parameter: learning rate (eta)
      
  Author: Annie Huang, 
"""
import mnist_loader
import network

# network.py example

#reads input data
training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data , 30, 10, 3.0, test_data=test_data)
