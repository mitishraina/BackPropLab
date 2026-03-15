# positional encoding: this gives tranformers the ability to understand token order
# without this, attention treats sentences like a bag of words
# self attention has no notion of sequence order, without positional information, both sequences would look identical to the model
# positional encoding injects token position information into embeddings with PE(pos+k), can be computed from linear combintaions of previous encodings. this helps model generalize to longer sequences
# final output = token embedding + positional encoding

import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_len: int=5000):
        self.d_model = d_model
        self.max_len = max_len
        
        self.encoding = self.build_encoding()
        
        
    def build_encoding(self):
        pe = np.zeros((self.max_len, self.d_model))
        
        positions = np.arange(self.max_len).reshape(-1, 1)
        
        div_term = np.exp(
            np.arange(0, self.d_model, 2) *
            -(np.log(10000.0) / self.d_model)
        )
        
        pe[:,0::2] = np.sin(positions * div_term)
        pe[:,1::2] = np.cos(positions * div_term)
        
        return pe
        # result shape: (max_len, d_model)
        
    def forward(self, X):
        """
        X shape: (batch_size, seq_len, d_model)
        """
        seq_len = X.shape[1]
        return X + self.encoding[:seq_len]
    
# example usage
# batch = 2
# seq_len = 6
# d_model = 16
# X = np.random.randn(batch, seq_len, d_model)
# pe = PositionalEncoding(d_model)
# output = pe.forward(X)
# print("Output shape:", output.shape)