# last small primitive in full transformer
# critical for stabilizing training in deep networks
# the problem is deep network suffer from internal covariate shift, the distribution of activations changes during training
# normalization helps: stabilize gradients, speed up training, improve convergence

import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        
        self.gamma = np.ones((1, d_model))
        self.beta = np.zeros((1, d_model))
        
    def forward(self, X):
        """
        X shape: (batch_size, seq_len, d_model)
        """
        mean = np.mean(X, axis=-1, keepdims=True)
        
        variance = np.var(X, axis=-1, keepdims=True)
        
        X_norm = (X - mean) / np.sqrt(variance + self.eps)
        
        output = self.gamma * X_norm + self.beta
        return output
    
# example usage
# batch = 2
# seq_len = 5
# d_model = 16

# X = np.random.randn(batch, seq_len, d_model)
# layer_norm = LayerNorm(d_model)
# output = layer_norm.forward(X)
# print("Output shape:", output.shape)