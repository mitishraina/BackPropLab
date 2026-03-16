# input -> encoder -> encoder -> encoder -> encoder -> output representation
# each layer will refine token representations
# lets say, we denote an encoder block by: Enc(x)
# then stacked becomes: H1 = Enc(X)
#                       H2 = Enc(H1)
#                       H3 = Enc(H2)
#         finally: Hn

# Typical Transformer setting: BERT base
# Layers(N) = 12
# Hidden size = 768
# Heads = 12
# FFN size = 3072


import numpy as np
from encoder import Encoder
from pos_encoding import PositionalEncoding

class TransformerEncoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_len: int=5000):
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.pos_encoding = PositionalEncoding(
            d_model,
            max_len
        )
        
        self.layers = [
            Encoder(
                d_model,
                num_heads,
                d_ff
            )
            for _ in range(num_layers)
        ]
        
    def forward(self, X, mask=None):
        """
        X shape: (batch, seq_len, d_model)
        """
        X = self.pos_encoding.forward(X)
        
        for layer in self.layers:
            X = layer.forward(X, mask)
        return X
        
# example usage
# batch = 2
# seq_len = 8
# d_model = 16
# num_heads = 4
# d_ff = 64
# num_layers = 3

# X = np.random.randn(batch, seq_len, d_model)
# encoder = TransformerEncoder(
#     num_layers,
#     d_model,
#     num_heads,
#     d_ff
# )
# output = encoder.forward(X)
# print("Output shape:", output.shape)

# What happens internally?
# Eg: The cat sat on the mat
# Layer1: Captures local relationships
#         cat <-> sat
#         sat <-> mat
# Layer2: Captures broader syntax:
        # cat <-> mat
# Layer3: Captures semantic structure:
    # subject -> verb -> object
# each layer builds higher level understanding

# Why stacking ?
# each encoder layer performs: 1. information mixing(attention)
#                              2. feature transformation(FFN)
# Stacking allows the network to build hierarchial representation