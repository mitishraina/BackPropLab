# BoW or Bag of Words
# text is converted into a "bag" of words, creating a document term matrix where rows are documents and columns are word counts
# includes tokenizing text, building a vocab of unique words, and creating vectors based on term frequency
# ignores word order and context, leading to loss of semantic meaning and often results in sparse, high dimensional data

import numpy as np
from collections import Counter
import re

class BagOfWords:
    def __init__(self, lower: bool=True, max_vocab_size=None):
        self.lower = lower
        self.max_vocab_size = max_vocab_size
        
        self.vocab = {}
        self.id_to_word = {}
        
    def tokenize(self, text):
        if self.lower:
            text = text.lower()
        
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
    
    def fit(self, corpus):
        counter = Counter()
        
        for doc in corpus:
            tokens = self.tokenize(doc)
            counter.update(tokens)
            
        most_common = counter.most_common(self.max_vocab_size)
        
        self.vocab = {
            word: idx for idx, (word, _) in enumerate(most_common)
        }
        
        self.id_to_word = {
            idx: word for word, idx in self.vocab.items()
        }
        
    def transform(self, corpus):
        N = len(corpus)
        V = len(self.vocab)
        
        X = np.zeros((N, V))
        
        for i, doc in enumerate(corpus):
            tokens = self.tokenize(doc)
            counts = Counter(tokens)
            
            for token, count in counts.items():
                if token in self.vocab:
                    j = self.vocab[token]
                    X[i, j] = count
        
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

# bow = BagOfWords()

# X = bow.fit_transform(corpus)

# print("Vocabulary:", bow.vocab)

# print("\nBoW Matrix:")
# print(X)