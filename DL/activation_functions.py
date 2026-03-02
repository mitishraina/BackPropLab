import numpy as np

class sigmoid:
    """
    Sigmoid saturates the vanishing gradients
    """
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self):
        return self.output * (1 - self.output)
    

class tanh:
    """
    tanh centers data and is better than sigmoid
    """
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self):
        return 1 - self.output ** 2
    
    
class ReLU:
    """
    ReLU prevents saturation(partially)
    """
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self):
        return (self.input > 0).astype(float)
    

class GELU:
    """
    GELU smooth ReLU which is better for deep networks
    """
    def forward(self, x):
        self.input = x
        return 0.5 * x * (
            1 + np.tanh(
                np.sqrt(2 / np.pi) *
                (x + 0.044715 * np.power(x, 3))
            )
        )
        
    def backward(self):
        x = self.input
        tanh_term = np.tanh(
            np.sqrt(2 / np.pi) *
            (x + 0.044715 * x ** 3)
        )
        
        sech2 = 1 - tanh_term ** 2
        
        term1 = 0.5 * (1 + tanh_term)
        term2 = 0.5 * x * sech2 * np.sqrt(2 / np.pi) * (
            1 + 3 * 0.044715 * x ** 2
        )
        
        return term1 + term2
    
    
class Softmax:
    """
    Softmax outputs probability distribution
    """
    def forward(self, x):
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(x_stable)
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.output
    
    def backward(self, d_out):
        """
        d_out: gradient from next layer
        returns gradient wrt input
        """
        batch_size, n_classes = self.output.shape
        dx = np.zeros_like(d_out)
        
        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            dx[i] = np.dot(jacobian, d_out[i])
    
        return dx