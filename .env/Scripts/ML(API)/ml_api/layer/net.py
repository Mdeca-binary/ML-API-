import numpy as np


class Network:
    
    def __init__(self, n_inputs:int, n_neurons:int):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.ones((1, n_neurons))
        
    def feedforward(self, inputs:list):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        