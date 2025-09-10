from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, input_param):
        self.input_param = input_param
        return np.dot(self.weights, self.input_param) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        pass