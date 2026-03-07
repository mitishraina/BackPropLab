# TFIDF vectorizer: term frequency inverset document frequency
# TF measures how often a word appears in a document
# idf measures how rare a word is across the corpus
# boost rare, signifacnt words and lowers the weight of common, less informative words
# improves upon BoW models by assigning higher weights to unique, meaningful words, making it essential for text mining and information retrieval
# limitation: still ignores word order, context and semantics

import numpy as np
from collections import Counter
import re

class TFIDF:
    def __init__(self, lower: bool=True, max_vocab_size=None):
        self.lower = lower
        self.max_vocab_size = max_vocab_size
        
        self.vocab = {}
        self.idf = None
        
    def tokenize(self, text):
        if self.lower:
            text = text.lower()
            
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
    
    def fit(self, corpus):
        N = len(corpus)
        
        df_counter = Counter()
        
        for doc in corpus:
            tokens = set(self.tokenize(doc))
            df_counter.update(tokens)
            
        most_common = df_counter.most_common(self.max_vocab_size)
        
        self.vocab = {
            word: idx for idx, (word, _) in enumerate(most_common)
        }
        
        self.idf = np.zeros(len(self.vocab))
        
        for word, idx in self.vocab.items():
            df = df_counter[word]
            
            self.idf[idx] = np.log(N / (1 + df)) + 1
            
    def transform(self, corpus):
        N = len(corpus)
        V = len(self.vocab)
        
        X = np.zeros((N, V))
        
        for i, doc in enumerate(corpus):
            tokens = self.tokenize(doc)
            counts = Counter(tokens)
            
            total_terms = len(tokens)
            
            for token, count in counts.items():
                if token in self.vocab:
                    j = self.vocab[token]
                    tf = count / total_terms
                    
                    X[i, j] = tf * self.idf[j]
                    
        return X
    
    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
    
    
    def vocab_size(self):
        return len(self.vocab)
    
# example usage
# corpus = [
#     "I love NLP",
#     "NLP is amazing",
#     "I love machine learning"
# ]

# vectorizer = TFIDF()

# X = vectorizer.fit_transform(corpus)

# print("Vocabulary:")
# print(vectorizer.vocab)

# print("\nTF-IDF Matrix:")
# print(X)