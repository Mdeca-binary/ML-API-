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
                
                
                
    