import numpy as np


class Layer:
    
    def __init__(self, n_inputs:int, n_neurons:int)->None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.ones((1, n_neurons))
        
    def forward(self, inputs:list)->None:
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
    
    def backward(self, derived_input:list)->None:
        self.derived_weights = np.dot(self.inputs.T, derived_input)
        self.derived_biases = np.sum(derived_input, axis=0, keepdims=True)
        self.derived_inputs = np.dot(derived_input, self.weights.T)

    