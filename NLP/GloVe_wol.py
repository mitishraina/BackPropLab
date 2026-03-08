# GloVe: global vectors for word representation
# creates word embeddings by analyzing co-occurence statistics across an entire corpus
# bridges matrix factorization with local context methods to capture deep semantic relationships

import numpy as np
import re
from collections import Counter, defaultdict

class GloVe:
    def __init__(self, embedding_dim: int=50, window_size: int=2, x_max: int=100, alpha: float=0.75, lr: int=0.05, epochs: int=25):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.x_max = x_max
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        
    def tokenize(self, text):
        text = text.lower()
        return re.findall(r"\b\w+\b", text)
    
    def build_vocab(self, corpus):
        tokens = []
        for text in corpus:
            tokens.extend(self.tokenize(text))
            
        counts = Counter(tokens)
        
        self.word2id = {w:i for i,(w,_) in enumerate(counts.itmes())}
        self.id2word = {i:w for w, i in self.word2id.items()}
        
        self.vocab_size = len(self.word2id)
        
    # this is the ccore of glove, co-occurence matrix
    def build_cooccurence(self, corpus):
        cooc = defaultdict(float)
        
        for text in corpus:
            tokens = self.tokenize(text)
            ids = [self.word2id[w] for w in tokens]
            
            for i, center in enumerate(ids):
                for j in range(
                    max(0, i-self.window_size),
                    min(len(ids), i+self.window_size+1)
                ):
                    if i != j:
                        context = ids[j]
                        distance = abs(i-j)
                        
                        cooc[(center, context)] += 1 / distance
                        
        return cooc
    
    def weighting(self, x):
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1
    
    def initialize_embeddings(self):
        self.W = np.random.randn(
            self.vocab_size,
            self.embedding_dim
        ) * 0.01
        
        self.W_tilde = np.random.randn(
            self.vocab_size,
            self.embedding_dim
        ) * 0.01
        
        self.b = np.zeros(self.vocab_size)
        self.b_tilde = np.zeros(self.vocab_size)
        
    def train(self, corpus):
        self.build_vocab(corpus)
        
        cooc = self.build_cooccurence(corpus)
        self.initialize_embeddings()
        for epoch in range(self.epochs):
            total_loss = 0
            for (i, j), Xij in cooc.items():
                w_i = self.W[i]
                w_j = self.W[j]
                
                weight = self.weighting(Xij)
                
                log_x = np.log(Xij)
                
                prediction = np.dot(w_i, w_j) + self.b[i] + self.b_tilde[j]
                
                error = prediction - log_x
                loss = weight * error**2
                
                total_loss += loss
                
                grad = weight * error
                
                self.W[i] -= self.lr * grad * w_j
                self.W_tilde -= self.lr * grad * w_i
                
                self.b[i] -= self.lr * grad
                self.b_tilde -= self.lr * grad
                
            print(f"epoch {epoch+1}, loss: {total_loss:.4f}")
            
    def get_embedding(self, word):
        idx = self.word2id[word]
        return self.W[idx] + self.W_tilde[idx]
    
# example usage
# corpus = [
#     "I love natural language processing",
#     "I love machine learning",
#     "machine learning loves data",
#     "natural language models are powerful"
# ]

# model = GloVe(
#     embedding_dim=20,
#     window_size=2,
#     epochs=50
# )

# model.train(corpus)

# print(model.get_embedding("machine"))