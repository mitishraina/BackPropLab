# Encoder: neural network component that processes entire input sequences in parallel to generate rich, contextualized numerical representations(embeddings)
# mainly performs two main operations:
    # 1. self attention: mix information across tokens
    # 2. feed forward network: transform each token independetly
# input -> multi-head attention -> add + layernorm -> ffn -> add + layernorm -> output

# step1: self attention: A = Multihead(X)
# step2: residual+norm: X_1 = LayerNorm(X+A)
# step3: feed forward: F = FFN(X_1)
# step4: residual_norm: Output = LayerNorm(X_1+F)

import numpy as np
from attention.multi_head_attention import MultiHeadAttention
from .feed_forward_network import FeedForward
from .layer_norm import LayerNorm

class Encoder:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        
        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )
        
        self.ffn = FeedForward(
            d_model,
            d_ff
        )
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
    def forward(self, X, mask=None):
        attn_output, _ = self.attention.forward(X, mask)
        
        X = self.norm1.forward(X + attn_output)
        
        ffn_output = self.ffn.forward(X)
        
        output = self.norm2.forward(X + ffn_output)
        
        return output
    
# example usage
# batch = 2
# seq_len = 6
# d_model = 16
# num_heads = 4
# d_ff = 64

# X = np.random.randn(batch, seq_len, d_model)
# encoder = Encoder(
#     d_model,
#     num_heads,
#     d_ff
# )
# output = encoder.forward(X)
# print("Output shape:", output.shape) # Output Shape: (2, 6, 16)

# What happens internally?
# Eg: The cat sat on the mat
# step1: attention: cat attends to -> the, sat, mat
# step2: residual keeps original information
# step3: ffn transforms token representations
