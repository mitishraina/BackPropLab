# word2vec: skipgram
# converts words into numerical vectors(word embeddings) using shallow neural networks.
# captures semantic relationships, similear words are place closed together in vector space
# for instance, model can understand "king" and "queen" are related, or "king - man + woman = queen"
# uses negative sampling to reduce computational load making it efficient


import numpy as np
import random
from collections import Counter
import re

class SkipGram:
    def __init__(self, embedding_dim: int=50, window_size: int=2, negative_samples: int=5, lr: int=0.01, epochs: int=5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
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
        
        self.word2id = {w: i for i, (w, _) in enumerate(counts.items())}
        self.id2word = {i: w for w, i in self.word2id.items()}
        
        self.vocab_size = len(self.word2id)
        
    def generate_pairs(self, corpus):
        pairs = []
        
        for text in corpus:
            tokens = self.tokenize(text)
            ids = [self.word2id[w] for w in tokens if w in self.word2id]
            
            for i, center in enumerate(ids):
                for j in range(
                    max(0, i-self.window_size),
                    min(len(ids), i+self.window_size+1)
                ):
                    if i != j:
                        pairs.append((center, ids[j]))
                        
        return pairs
    
    def initialize_embeddings(self):
        self.W_in = np.random.randn(
            self.vocab_size,
            self.embedding_dim
        ) * 0.01
        
        self.W_out = np.random.randn(
            self.vocab_size,
            self.embedding_dim
        ) * 0.01
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sample_negative(self, target):
        negatives = []
        
        while len(negatives) < self.negative_samples:
            neg = random.randn(0, self.vocab_size-1)
            
            if neg != target:
                negatives.append(neg)
                
        return negatives
    
    def train(self, corpus):
        self.build_vocab(corpus)
        
        pairs = self.generate_pairs(corpus)
        
        self.initialize_embeddings()
        
        for epoch in range(self.epochs):
            loss = 0
            for center, context in pairs:
                v_c = self.W_in[center]
                v_o = self.W_out[context]
                
                score = np.dot(v_c, v_o)
                pos_prob = self.sigmoid(score)
                
                loss += -np.log(pos_prob)
                grad = pos_prob - 1
                
                self.W_in[center] -= self.lr * grad * v_o
                self.W_out[center] -= self.lr * grad * v_c
                
                negatives = self.sample_negative(context)
                
                for neg in negatives:
                    v_n = self.W_out[neg]
                    score = np.dot(v_c, v_n)
                    neg_prob = self.sigmoid(score)
                    
                    loss += -np.log(1 - neg_prob)
                    
                    grad = neg_prob
                    
                    self.W_in[center] -= self.lr * grad * v_n
                    self.W_out[neg] -= self.lr * grad * v_c
                    
            print(f"Epoch {epoch+1}, loss: {loss:.4f}")
            
    def get_embedding(self, word):
        idx = self.word2id[word]
        return self.W_in[idx]
    
# example usage
# corpus = [
#     "I love natural language processing",
#     "I love machine learning",
#     "machine learning loves data",
#     "natural language models are powerful"
# ]

# model = SkipGram(
#     embedding_dim=20,
#     window_size=2,
#     negative_samples=4,
#     epochs=50
# )

# model.train(corpus)

# print(model.get_embedding("machine"))