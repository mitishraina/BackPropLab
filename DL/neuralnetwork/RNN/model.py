# this is a vanilla RNN (many to one)
# A class of ANN designed to process sequential data (text, speect, time series) by retaining a hidden state or memory of previous inputs

import numpy as np

class RNN:
    def __init__(self, input_size, output_size, hidden_size, lr: int=0.01, seed: int=42):
        np.random.seed(seed)
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lr = lr
        
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
    
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        
        self.X = X
        self.h = {}
        self.h[-1] = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            
            self.h[t] = np.tanh(
                x_t @ self.W_xh + self.h[t-1] @ self.W_hh + self.b_h
            )
            
        self.y_logits = self.h[seq_len-1] @ self.W_hy + self.b_y
        
        return self.y_logits
    
    def compute_loss(self, logits, y):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        
        N = logits.shape[0]
        loss = -np.mean(
            np.log(self.probs[np.arange(N), y])
        )
        return loss
    
    def backward(self, y):
        batch_size, seq_len, _ = self.X.shape
        
        # gradients initialization
        dw_xh = np.zeros_like(self.W_xh)
        dw_hh = np.zeros_like(self.W_hh)
        dw_hy = np.zeros_like(self.W_hy)
        
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # output gradient
        d_logits = self.probs.copy()
        d_logits[np.arange(batch_size), y] -= 1
        d_logits /= batch_size
        
        # gradients for output layer
        dw_hy += self.h[seq_len-1].T @ d_logits
        db_y += np.sum(d_logits, axis=0, keepdims=True)
        
        # backprop inot last hidden state
        dh_next = d_logits @ self.W_hy.T
        
        # backprop through time
        for t in reversed(range(seq_len)):
            dh = dh_next
            
            dtanh = dh * (1 - self.h[t] ** 2)
            
            dw_xh += self.X[:, t, :].T @ dtanh
            dw_hh += self.h[t-1].T @ dtanh
            db_h += np.sum(dtanh, axis=0, keepdims=True)
            
            dh_next = dtanh @ self.W_hh.T
            
        # udpate parameters
        self.W_xh -= self.lr * dw_xh
        self.W_hh -= self.lr * dw_hh
        self.W_hy -= self.lr * dw_hy
        
        self.b_h -= self.lr * db_h
        self.b_y -= self.lr * db_y
        
        
# RNN can learn sequential dependencies
# longer sequence -> vanishing gradients
# gradient norms may explode (try increasing seq_len)
# limitations of Vanilla RNN
# 1. Vanishing gradients
# 2. exploding gradients
# 3. long term memory issue
# 4. sequential computation