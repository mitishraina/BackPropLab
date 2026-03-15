# multi head attention is the mechanism that allows transformers to learn different types of relationships
# a single head attention can only learn one type of relationship, so transformers use mulitple attention heads in parallel
# input embeddings -> linear projections(Q,K,v) -> split into multiple heads -> sclaed dot product (parallel) -> concatenate outputs -> final linear projection

import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def scaled_dot_product(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    
    scores = np.matmul(Q, K.transpose(0,2,1))
    scores = scores / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)
    
    weights = softmax(scores)
    output = np.matmul(weights, V)
    return output, weights

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
    def split_heads(self, X):
        batch, seq_len, d_model = X.shape
        
        X = X.reshape(batch, seq_len, self.num_heads, self.d_k)
        return X.transpose(0,2,1,3)
        # result shape: (batch, heads, seq_len, d_k)
    
    def combine_heads(self, X):
        batch, heads, seq_len, d_k = X.shape
        X = X.transpose(0,2,1,3)
        
        return X.reshape(batch, seq_len, heads*d_k)
    
    def forward(self, X, mask=None):
        batch, seq_len, _  = X.shape
        
        # linear projections
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        
        # split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        outputs = []
        attn_weights = []
        
        for h in range(self.num_heads):
            out, w = scaled_dot_product(
                Q[:,h],
                K[:,h],
                V[:,h],
                mask
            )
            
            outputs.append(out)
            attn_weights.append(w)
            
        outputs = np.stack(outputs, axis=1)
        
        concat = self.combine_heads(outputs)
        
        output = concat @ self.W_o
        
        return output, attn_weights
    
# example usage
# batch = 2
# seq_len = 5
# d_model = 16
# num_heads = 4

# X = np.random.randn(batch, seq_len, d_model)

# mha = MultiHeadAttention(d_model, num_heads)

# output, weights = mha.forward(X)

# print("Output shape:", output.shape)

# this will split head into 4 heads as (head1, head2, head3, head4) attention then concatenate it