# gru or gated recurrent unit, a simplification of LSTM
# it keeps gating mechanism, stable gradients, long term dependency but removes separate cell state, one gate, and some parameter overhead


import numpy as np

class GRU:
    def __init__(self, input_size, hidden_size, output_size, lr: int=0.01, seed: int=42):
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        
        def init_gate():
            return (
                np.random.randn(input_size, hidden_size) * 0.01,
                np.random.randn(hidden_size, hidden_size) * 0.01,
                np.zeros((1, hidden_size))
            )
            
        self.W_z, self.U_z, self.b_z = init_gate()
        self.W_r, self.U_r, self.b_r = init_gate()
        self.W_h, self.U_h, self.b_h = init_gate()
        
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        self.b_y = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        
        self.X = X
        self.h = {}
        self.h[-1] = np.zeros((batch_size, self.hidden_size))
        
        self.cache = []
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            h_prev = self.h[t-1]
            
            z = self.sigmoid(x_t @ self.W_z + h_prev @ self.U_z + self.b_z)
            r = self.sigmoid(x_t @ self.W_r + h_prev @ self.U_r + self.b_r)
            
            h_tilde = self.tanh(
                x_t @ self.W_h +
                (r * h_prev) @ self.U_h +
                self.b_h
            )
            
            h = (1 - z) * h_prev + z * h_tilde
            
            self.h[t] = h
            self.cache.append((z, r, h_tilde, h_prev))
            
        logits = self.h[seq_len-1] @ self.W_hy + self.b_y
        return logits
    
    def compute_loss(self, logits, y):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        
        N = logits.shape[0]
        loss = -np.mean(np.log(self.probs[np.arange(N), y]))
        return loss
    
    def backward(self, y):
        batch_size, seq_len, _ = self.X.shape

        # Initialize gradients
        dW_z = np.zeros_like(self.W_z)
        dU_z = np.zeros_like(self.U_z)
        db_z = np.zeros_like(self.b_z)

        dW_r = np.zeros_like(self.W_r)
        dU_r = np.zeros_like(self.U_r)
        db_r = np.zeros_like(self.b_r)

        dW_h = np.zeros_like(self.W_h)
        dU_h = np.zeros_like(self.U_h)
        db_h = np.zeros_like(self.b_h)

        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        # Output gradient
        d_logits = self.probs.copy()
        d_logits[np.arange(batch_size), y] -= 1
        d_logits /= batch_size

        dW_hy += self.h[seq_len-1].T @ d_logits
        db_y += np.sum(d_logits, axis=0, keepdims=True)

        dh_next = d_logits @ self.W_hy.T

        for t in reversed(range(seq_len)):
            z, r, h_tilde, h_prev = self.cache[t]
            h = self.h[t]

            dh = dh_next

            # h = (1 - z)*h_prev + z*h_tilde
            dh_tilde = dh * z
            dz = dh * (h_tilde - h_prev)
            dh_prev = dh * (1 - z)

            # h_tilde = tanh(...)
            dh_tilde_raw = dh_tilde * (1 - h_tilde**2)

            x_t = self.X[:, t, :]

            dW_h += x_t.T @ dh_tilde_raw
            dU_h += (r * h_prev).T @ dh_tilde_raw
            db_h += np.sum(dh_tilde_raw, axis=0, keepdims=True)

            # Reset gate
            dr_part = dh_tilde_raw @ self.U_h.T
            dr = dr_part * h_prev
            dr_raw = dr * r * (1 - r)

            dW_r += x_t.T @ dr_raw
            dU_r += h_prev.T @ dr_raw
            db_r += np.sum(dr_raw, axis=0, keepdims=True)

            # Update gate
            dz_raw = dz * z * (1 - z)

            dW_z += x_t.T @ dz_raw
            dU_z += h_prev.T @ dz_raw
            db_z += np.sum(dz_raw, axis=0, keepdims=True)

            dh_prev += (
                dh_tilde_raw @ self.U_h.T * r +
                dr_raw @ self.U_r.T +
                dz_raw @ self.U_z.T
            )

            dh_next = dh_prev

        # Update parameters
        for param, grad in zip(
            [self.W_z, self.U_z, self.b_z,
             self.W_r, self.U_r, self.b_r,
             self.W_h, self.U_h, self.b_h,
             self.W_hy, self.b_y],
            [dW_z, dU_z, db_z,
             dW_r, dU_r, db_r,
             dW_h, dU_h, db_h,
             dW_hy, db_y]
        ):
            param -= self.lr * grad