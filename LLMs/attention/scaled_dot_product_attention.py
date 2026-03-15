# scaled dot product attention
# instead of processing tokens sequentially like RNNs, this allows each token to look at every other token
# this attention computes relevance scores between tokens
# input embeddings -> linear projections -> queries, keys, values(Q,K,V) -> attention scores = QK_T -> sclae by root dk -> softmax -> weighted sum of V

import numpy as np

def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    exp = np.exp(X)
    
    return exp / np.sum(exp, axis=-1, keepdims=True)

class ScaledDotProduct:
    """
        Q shape: (seq_len_q, d_k)
        K shape: (seq_len_k, d_k)
        V shape: (seq_len_k, d_v)
    """
    def __init__(self):
        pass
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = np.dot(Q, K.T)
        
        scores = scores / np.sqrt(d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
            
        attention_weights = softmax(scores)
        output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
# each row corresponds to one token attending to all tokens
# also, need to implement scaling here because without scaling: QK_T, so attention becomes very large for large dimensions -> softmax saturates -> gradients vanish
# scaling fixes this: QK_T / root(d_k)