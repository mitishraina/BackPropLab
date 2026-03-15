# seq2seq or sequence to sequence model is a type of neural network architecture desinged to convert input sequences into output sequences
# used for tasks where both input and output are variable length sequences, such as machine translation(google translator), text summarization, chatbot responses
# Input sentence -> Encoder -> Context Vector -> Decoder -> Output
import numpy as np

class Encoder:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(input_size, hidden_size) * 0.01
        self.b_h = np.zeros((1, hidden_size))
        
    def forward(self, X):
        seq_len = X.shape[0]
        self.h = {}
        
        self.h[-1] = np.zeros((1, self.hidden_size))
        
        for t in range(seq_len):
            x_t = X[t].reshape(1, -1)
            self.h[t] = np.tanh(
                x_t @ self.W_xh +
                self.h[t-1] @ self.W_hh +
                self.b_h
            )
        
        return self.h[seq_len-1]
    
class Decoder:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
    def forward(self, Y, context):
        seq_len = Y.shape[0]
        self.s = {}
        self.s[-1] = context
        
        outputs = []
        
        for t in range(seq_len):
            y_t = Y[t].reshape(1, -1)
            self.s[t] = np.tanh(
                y_t @ self.W_xh +
                self.s[t-1] @ self.W_hh +
                self.b_h
            )
            
            logits = self.s[t] @ self.W_hy + self.b_y
            outputs.append(logits)
    
        return outputs
    
class seq2seq:
    def __init__(self, input_size, hidden_size, output_size):
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size, output_size)
        
    def forward(self, X, Y):
        context = self.encoder.forward(X)
        outputs = self.decoder.forward(Y, context)
        
        return outputs
    
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
    

# Limitations of seq2seq
# Context vector bottleneck. The encoder compresses entire sentence into one vector, for long sequences this fails.
# TOo much informations -> lost in compression
# this led to "Bahdanau Attention": instead of one context vector, decoder attends to all encoder states
