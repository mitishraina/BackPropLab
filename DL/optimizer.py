import numpy as np

class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
        
    def step(self, grads):
        raise NotImplementedError
    
    
class GradientDescent(Optimizer):
    def step(self, grads):
        for p, g in zip(self.params, grads):
            p -= self.lr * g
            
            
class SGD(Optimizer):
    """
    Same update Rule as GD, difference is how gradients are computed
    """
    def step(self, grads):
        for p, g in zip(self.params, grads):
            p -= self.lr * g
            
            
class AdaGrad(Optimizer):
    def __init__(self, params, lr=0.01, epsilon=1e-8):
        super().__init__(params, lr)
        self.epsilon = epsilon
        self.G = [np.zeros_like(p) for p in self.params]
        
    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.G[i] += g ** 2
            adjusted_lr = self.lr / (np.sqrt(self.G[i]) + self.epsilon)
            p -= adjusted_lr * g
            
            
class RMSProp(Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(params, lr)
        self.beta = beta
        self.epsilon = epsilon
        self.Eg = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.Eg[i] = (
                self.beta * self.Eg[i] +
                (1 - self.beta) * (g ** 2)
            )

            p -= self.lr * g / (np.sqrt(self.Eg[i]) + self.epsilon)
            
            
class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01
    ):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1

        for i, (p, g) in enumerate(zip(self.params, grads)):
            # Update biased first moment
            self.m[i] = (
                self.beta1 * self.m[i] +
                (1 - self.beta1) * g
            )

            # Update biased second moment
            self.v[i] = (
                self.beta2 * self.v[i] +
                (1 - self.beta2) * (g ** 2)
            )

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Parameter update
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Decoupled weight decay
            p -= self.lr * self.weight_decay * p