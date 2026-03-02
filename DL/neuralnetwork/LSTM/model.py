# LSTM: long short term memory
# desinged to leran long term dependencies in sequential data, overcoming vanishing gradient problem of standard RNNs
# uses internal mechanisms called gates to regulate information flow
# effectively store, update and discard information, making them ideal for time series forecasting, NLP and speech recognition
# forget, input, candidate, cell update, output, hidden state

import numpy as np

class LSTM:
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
            
        self.W_f, self.U_f, self.b_f = init_gate()
        self.W_i, self.U_i, self.b_i = init_gate()
        self.W_c, self.U_c, self.b_c = init_gate()
        self.W_o, self.U_o, self.b_o = init_gate()
        
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
        self.c = {}
        
        self.h[-1] = np.zeros((batch_size, self.hidden_size))
        self.c[-1] = np.zeros((batch_size, self.hidden_size))
        
        self.cache = []
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            h_prev = self.h[t-1]
            c_prev = self.c[t-1]
            
            f = self.sigmoid(x_t @ self.W_f + h_prev @ self.U_f + self.b_f)
            i = self.sigmoid(x_t @ self.W_i + h_prev @ self.U_i, self.b_i)
            c_tilde = self.tanh(x_t @ self.W_c + h_prev @ self.U_c + self.b_c)
            
            c = f * c_prev + i * c_tilde
            o = self.sigmoid(x_t @ self.W_o + h_prev @ self.U_o + self.b_o)
            
            h = o * self.tanh(c)
            
            self.h[t] = h
            self.c[t] = c
            
            self.cache.append((f, i, c_tilde, o, c_prev))
            
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
        dW_f = np.zeros_like(self.W_f)
        dU_f = np.zeros_like(self.U_f)
        db_f = np.zeros_like(self.b_f)

        dW_i = np.zeros_like(self.W_i)
        dU_i = np.zeros_like(self.U_i)
        db_i = np.zeros_like(self.b_i)

        dW_c = np.zeros_like(self.W_c)
        dU_c = np.zeros_like(self.U_c)
        db_c = np.zeros_like(self.b_c)

        dW_o = np.zeros_like(self.W_o)
        dU_o = np.zeros_like(self.U_o)
        db_o = np.zeros_like(self.b_o)

        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        # Output gradient
        d_logits = self.probs.copy()
        d_logits[np.arange(batch_size), y] -= 1
        d_logits /= batch_size

        dW_hy += self.h[seq_len-1].T @ d_logits
        db_y += np.sum(d_logits, axis=0, keepdims=True)

        dh_next = d_logits @ self.W_hy.T
        dc_next = np.zeros_like(self.c[0])

        # Backprop through time
        for t in reversed(range(seq_len)):
            f, i, c_tilde, o, c_prev = self.cache[t]

            h = self.h[t]
            c = self.c[t]

            dh = dh_next
            do = dh * np.tanh(c)
            do_raw = do * o * (1 - o)

            dc = dh * o * (1 - np.tanh(c)**2) + dc_next
            dc_tilde = dc * i
            dc_tilde_raw = dc_tilde * (1 - c_tilde**2)

            di = dc * c_tilde
            di_raw = di * i * (1 - i)

            df = dc * c_prev
            df_raw = df * f * (1 - f)

            x_t = self.X[:, t, :]
            h_prev = self.h[t-1]

            dW_f += x_t.T @ df_raw
            dU_f += h_prev.T @ df_raw
            db_f += np.sum(df_raw, axis=0, keepdims=True)

            dW_i += x_t.T @ di_raw
            dU_i += h_prev.T @ di_raw
            db_i += np.sum(di_raw, axis=0, keepdims=True)

            dW_c += x_t.T @ dc_tilde_raw
            dU_c += h_prev.T @ dc_tilde_raw
            db_c += np.sum(dc_tilde_raw, axis=0, keepdims=True)

            dW_o += x_t.T @ do_raw
            dU_o += h_prev.T @ do_raw
            db_o += np.sum(do_raw, axis=0, keepdims=True)

            dh_next = (
                df_raw @ self.U_f.T +
                di_raw @ self.U_i.T +
                dc_tilde_raw @ self.U_c.T +
                do_raw @ self.U_o.T
            )

            dc_next = dc * f

        # Update parameters
        for param, grad in zip(
            [self.W_f, self.U_f, self.b_f,
             self.W_i, self.U_i, self.b_i,
             self.W_c, self.U_c, self.b_c,
             self.W_o, self.U_o, self.b_o,
             self.W_hy, self.b_y],
            [dW_f, dU_f, db_f,
             dW_i, dU_i, db_i,
             dW_c, dU_c, db_c,
             dW_o, dU_o, db_o,
             dW_hy, db_y]
        ):
            param -= self.lr * grad