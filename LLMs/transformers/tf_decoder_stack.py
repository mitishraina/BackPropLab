# input -> positional encoding -> decoder block -> decoder block -> decoder block -> output representation
# each layer refines the generated sequence representation
# if a decoder block is: Dec(x)
# then stacked decoder:
#         H1 = Dec(X)
#         H2 = Dec(H1)
#         H3 = Dec(H2)
# Final: Hn

import numpy as np
from decoder import Decoder
from pos_encoding import PositionalEncoding

class TransformerDecoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_len: int=5000):
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.pos_encoding = PositionalEncoding(
            d_model,
            d_ff
        )
        
        self.layers = [
            Decoder(
                d_model,
                num_heads,
                d_ff
            )
            for _ in range(num_layers)
        ]
        
    def forward(self, X, encoder_output, mask=None):
        """
        X shape: (batch, seq_len, d_model)
        """
        
        X = self.pos_encoding.forward(X)
        
        for layer in self.layers:
            X = layer.forward(
                X, encoder_output, mask
            )
        
        return X
        
# example usage
# batch = 2
# seq_len = 6
# enc_seq_len = 8
# d_model = 16
# num_heads = 4
# d_ff = 64
# num_layers = 3

# X = np.random.randn(batch, seq_len, d_model)
# encoder_output = np.random.randn(
#     batch,
#     enc_seq_len,
#     d_model
# )
# decoder = TransformerDecoder(
#     num_layers,
#     d_model,
#     num_heads,
#     d_ff
# )
# output = decoder.forward(
#     X,
#     encoder_output
# )
# print("Output shape:", output.shape)

# output: (2,6,16)
# each decoder layer: 
    # 1. uses masked attention to see previous tokens
    # 2. uses cross attention to read encoder output
    # 3. applies ffn transformation