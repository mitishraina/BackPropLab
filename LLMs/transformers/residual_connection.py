# deep networks suffer from: vanishing gradients, exploding gradients, degradation of training accuracy
# residual connections solve this by letting gradients flow directly through the network
# instead of learning: y = F(x), the network learns: y = x+F(x), so the model learns a residual correction to the input
# residual: y = x + Sublayer(x)
# and in tranformers it is combined with normalization:
    # y = LayerNorm(x+ Sublayer(x))

import numpy as np

class ResidualConnection:
    def __init__(self, layer_norm):
        self.layer_norm = layer_norm
    
    def forward(self, X, sublayer_output):
        """
        X shape: (batch, seq_len, d_model)
        sublayer_output: output of attention or ffn
        """
        
        residual = X + sublayer_output
        return self.layer_norm.forward(residual)
    
# example usage
# batch = 2
# seq_len = 4
# d_model = 16

# X = np.random.randn(batch, seq_len, d_model)
# sublayer_output = np.random.randn(batch, seq_len, d_model)

# from layer_norm import LayerNorm
# layer_norm = LayerNorm(d_model)
# residual = ResidualConnection(layer_norm)
# output = residual.forward(X, sublayer_output)
# print("Output shape:", output.shape)
