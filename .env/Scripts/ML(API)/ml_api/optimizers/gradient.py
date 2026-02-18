import numpy as np

class StochasticGradientDescent:
    
    def __init__(self, lr=0.01, decay=0., momentum=0.):
        self.learning_rate = lr
        self.current_lr = lr
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0
        
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1. / (1. + self.decay * self.iteration))
                
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                    self.current_lr * layer.weights
            layer.weight_momentums = weight_updates
            bias_update = \
                self.momentum * layer.biases_momentums - \
                    self.current_lr * layer.dbiases
            layer.biases_momentums = bias_update
        else:
            weight_updates = -self.current_lr * \
                layer.dweights
            bias_update = -self.current_lr * \
                layer.dbiases
    
    def post_update_params(self):
        self.iteration = self.iteration + 1
    
class AdaGrad:
    
    def __init__(self, learning_rate=0.01, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay 
        self.epsilon = epsilon
        self.iteration = 0
        
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1. / (1. + self.decay * self.iteration))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights ** 2
        layer.biases_cache += layer.dbiases ** 2
        layer.weights += self.current_lr * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias_cache += self.current_lr * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iteration = self.iteration + 1

class RMSprop:
    
    def __init__(self, learning_rate=0.01, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iteration = 0
    
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1. / (1. + self.decay * self.iteration))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weights_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases ** 2
        layer.weights += -self.current_lr * \
            layer.dweights / \
                (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_lr * \
            layer.dbiases / \
                (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iteration = self.iteration + 1

class Adam:
    
    def __init__(self, learning_rate=0.01, decay=0., epsilon=1e-7, 
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iteration = 0
        
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1. / (1. + self.decay * self.iteration))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iteration + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iteration + 1))
        layer.weight_cache = self.beta_2 * layer.weights_cache + \
            (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases ** 2
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iteration + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iteration + 1))
        layer.weights += -self.current_lr * \
            weight_cache_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_lr * \
            bias_cache_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_update_params(self):
        self.iteration = self.iteration + 1
