# FFN: feed forward network is a 2 layer MLP applied to every token via:
# FFN(x)=max(0,xW1​+b1​)W2​+b2​, where w1 expands dimension and w2 projects back to model dimension


import numpy as np

def relu(x):
    return np.maximum(0, x)

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))
        
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))
        
    def forward(self, X):
        """
        X shape: (batch_size, seq_len, d_model)
        """
        batch, seq_len, _ = X.shape
        
        # flatten tokens for matrix multiplication
        X_flat = X.reshape(-1, self.d_model)
        
        hidden = relu(X_flat @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        
        output = output.reshape(batch, seq_len, self.d_model)
        
        return output
    
# example usage
# batch = 2
# seq_len = 5
# d_model = 16
# d_ff = 64

# X = np.random.randn(batch, seq_len, d_model)
# ffn = FeedForward(d_model, d_ff)
# output = ffn.forward(X)
# print("Output shape:", output.shape)