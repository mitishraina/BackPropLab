# this is full transformers model: encoder-decoder block

import numpy as np
from tf_encoder_stacked import TransformerEncoder
from tf_decoder_stacked import TransformerDecoder

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    
    exp = np.exp(x)
    
    return exp / np.sum(exp, axis=-1, keepdims=True)

class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len: int=5000):
        self.d_model = d_model
        
        self.src_embedding = np.random.randn(
            src_vocab_size,
            d_model
        ) * 0.01
        
        self.tgt_embedding = np.random.randn(
            tgt_vocab_size,
            d_model
        ) * 0.01
        
        self.encoder = TransformerEncoder(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            max_len
        )
        
        self.decoder = TransformerDecoder(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            max_len
        )
        
        self.W_out = np.random.randn(
            d_model,
            tgt_vocab_size
        ) * 0.01
        
    def embed(self, tokens, embedding_matrix):
        batch, seq_len = tokens.shape
        embeddings = embedding_matrix[tokens]
        return embeddings
    
    def forward(self, src_tokens, tgt_tokens, mask=None):
        src = self.embed(
            src_tokens,
            self.src_embedding
        )
        
        encoder_output = self.encoder.forward(
            src, 
            mask
        )
        
        tgt = self.embed(
            tgt_tokens,
            self.tgt_embedding
        )
        
        decoder_output = self.decoder.forward(
            tgt,
            encoder_output,
            mask
        )
        
        logits = decoder_output @ self.W_out
        
        probs = softmax(logits)
        return probs
    
# encoder learns representation
# decoder generates at each step which attends to:
    # 1. previous generated tokens
    # 2. encoder representation