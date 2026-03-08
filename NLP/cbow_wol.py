# CBOW: Continuous bag of word
# predicts a target word given the context of the surroundings word. 
# it does this by using a neural network with a sinlge hidden layer to learn the weights that map the context words to the target word.

import numpy as np
import re
import random
from collections import Counter

class CBOW:
    def __init__(self, embedding_dim: int=50, window_size: int=2, negative_sample: int=5, lr: int=0.01, epochs: int=5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_sample = negative_sample
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
        self.word2id = {w:i for i,(w, _) in enumerate(counts.items())}
        self.id2word = {i:w for w, i in self.word2id.items()}
        
        self.vocab_size = len(self.word2id)
        
    def generate_samples(self, corpus):
        samples = []
        
        for text in corpus:
            tokens = self.tokenize(text)
            ids = [self.word2id[w] for w in tokens]
            
            for i in range(len(ids)):
                context = []
                
                for j in range(
                    max(0, i-self.window_size),
                    min(len(ids), i+self.window_size+1)
                ):
                    if i != j:
                        context.append(ids[j])
                target = ids[i]
                
                samples.append((context, target))
                
        return samples
    
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
        
        while len(negatives) < self.negative_sample:
            neg = random.randint(0, self.vocab_size-1)
            if neg != target:
                negatives.append(neg)
                
        return negatives
    
    
    def train(self, corpus):
        self.build_vocab(corpus)
        
        samples = self.generate_samples(corpus)
        self.initialize_embeddings()
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            for context, target in samples:
                context_vectors = self.W_in[context]
                v_c = np.mean(context_vectors, axis=0)
                
                v_o = self.W_out[target]
                
                score = np.dot(v_c, v_o)
                pos_prob = self.sigmoid(score)
                
                loss = -np.log(pos_prob)
                total_loss += loss
                
                grad = pos_prob - 1
                self.W_out[target] -= self.lr * grad * v_c
                
                for word in context:
                    self.W_in[word] -= self.lr * grad * v_o / len(context)
                    
                negatives = self.sample_negative(target)
                
                for neg in negatives:
                    v_n = self.W_out[neg]
                    
                    score = np.dot(v_c, v_n)
                    neg_prob = self.sigmoid(score)
                    
                    loss = -np.log(1 - neg_prob)
                    total_loss += loss
                    
                    grad = neg_prob
                    
                    self.W_out[neg] -= self.lr * grad * v_c
                    
                    for word in context:
                        self.W_in[word] -= self.lr * grad * v_n / len(context)
                        
                print(f"epoch {epoch+1}, loss: {total_loss:.4f}")
                
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

# model = CBOW(
#     embedding_dim=20,
#     window_size=2,
#     negative_samples=4,
#     epochs=50
# )

# model.train(corpus)

# print(model.get_embedding("machine"))