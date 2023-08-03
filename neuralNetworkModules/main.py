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
import network2

# network.py example

#reads input data
training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data , 30, 10, 3.0, test_data=test_data)


#network2.py example1
training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
net2 =network2.Network([784, 30, 10], cost= network2.CrossEntropyCost)
net2.large_weight_initializer()
net2.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,monitor_evaluation_accuracy=True)

#network2.py example2
training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
net2a = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
