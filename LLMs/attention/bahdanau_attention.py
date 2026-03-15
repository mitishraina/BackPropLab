# Instead of using only one final encoder state, allow the decoder to look at all encoder states.
# Encoder states:
# h1 h2 h3 h4 h5
#  ↓  ↓  ↓  ↓  ↓
# Decoder attends to relevant words
# for each output step, the model learns where to focus
# Architecture:
# Encoder:
# x1 → h1
# x2 → h2
# x3 → h3
# x4 → h4
# Decoder step t:
# scores = alignment(s_{t-1}, h_i)
# α = softmax(scores)
# context = Σ α_i h_i
# decoder_input = [y_{t-1}, context]

# step1: alignment score
# step2: attention weights
# step3: context vector

import numpy as np

def softmax(X):
    exp = np.exp(X - np.max(X))
    return exp / np.sum(exp)

class BahdanauAttention:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_s = np.random.randn(hidden_size, hidden_size) * 0.01
        
        self.v = np.random.randn(hidden_size, 1) * 0.01
        
    def score(self, decoder_state, encoder_state):
        score = np.tanh(
            encoder_state @ self.W_h +
            decoder_state @ self.W_s
        )
        
        score = score @ self.v
        return score
    
    def compute_context(self, decoder_state, encoder_state):
        scores = []
        
        for h_i in encoder_state:
            e = self.score(decoder_state, h_i)
            scores.append(e)
            
        scores = np.array(scores).reshape(-1)
        attention_weights = softmax(scores)
        context = np.zeros((1, self.hidden_size))
        
        for i, h_i in enumerate(encoder_state):
            context += attention_weights[i] * h_i
        
        return context, attention_weights
    
# oldest attention released in 2014 which is replaced with Scaled Dot Product attention now(modern day attention used in transformers)