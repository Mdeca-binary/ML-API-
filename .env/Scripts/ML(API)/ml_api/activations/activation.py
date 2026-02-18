import numpy as np


class ReLU:
    
    def feedforward(self, inputs:list):
        self.output = np.maximum(0, inputs)
        
class SoftMax:
    
    def feedforward(self, inputs:list):
        exp_val = np.exp(inputs - np.exp(inputs, axis=1, keepdims=True))
        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = probabilities
        
class Sigmoid:
    
    def feedforward(self, inputs:list):
        self.output = 1. / (1 + np.exp(-inputs))
        
class LeakyReLU:
    
    def feedforward(self, alpha:float, inputs:list):
        self.output = np.where(inputs > 0, inputs, \
                               np.multiply(alpha, inputs))
        
class Elu:
    
    def feedforward(self, alpha:float, inputs:list):
        self.output = np.where(inputs > 0, inputs, \
                        np.multiply(alpha, np.exp(inputs) - 1))