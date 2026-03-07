import re
from collections import Counter

class Tokenizer:
    def __init__(self, lower: bool=True, max_vocab_size=None, min_freq: int=1):
        self.lower = lower
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        self.token_to_id = {}
        self.id_to_token = {}
        
    def normalize(self, text):
        if self.lower:
            text = text.lower()
            
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    
    def tokenize(self, text):
        text = self.normalize(text)
        tokens = re.findall(r"\b\w+\b", text)
        
        return tokens
    
    def build_vocab(self, texts):
        counter = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
            
        vocab = []
        
        for token, freq in counter.items():
            if freq >= self.min_freq:
                vocab.append((token, freq))
                
        vocab.sort(key=lambda x: x[1], reverse=True)
        
        if self.max_vocab_size:
            vocab = vocab[:self.max_vocab_size]
            
        tokens = self.special_tokens + [t[0] for t in vocab]
        
        self.token_to_id = {
            token: idx for idx, token in enumerate(tokens)
        }
        
        self.id_to_token = {
            idx: token for token, idx in self.token_to_id.items()
        }
        
    def encode(self, text, add_special_tokens: bool= True, max_length=None, padding: bool=False):
        tokens = self.tokenize(text)
        
        ids = []
        
        if add_special_tokens:
            ids.append(self.token_to_id[self.bos_token])
            
        for token in tokens:
            ids.append(
                self.token_to_id.get(
                    token,
                    self.token_to_id[self.unk_token]
                )
            )
            
        if add_special_tokens:
            ids.append(self.token_to_id[self.eos_token])
            
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
                
            if padding:
                pad_id = self.token_to_id[self.pad_token]
                ids += [pad_id] * (max_length - len(ids))
                
        return ids
    
    def decode(self, ids, skip_special_tokens: bool=True):
        tokens = []
        
        for idx in ids:
            token = self.id_to_token.get(idx, self.unk_token)
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            token.append(token)
            
        return " ".join(tokens)
    
    def vocab_size(self):
        return len(self.token_to_id)
    
# example usage
# texts = [
#     "I love NLP",
#     "NLP is amazing",
#     "I love machine learning"
# ]

# tokenizer = Tokenizer()

# tokenizer.build_vocab(texts)

# print("Vocab size:", tokenizer.vocab_size())

# encoded = tokenizer.encode(
#     "I love NLP",
#     max_length=10,
#     padding=True
# )

# print("Encoded:", encoded)

# decoded = tokenizer.decode(encoded)

# print("Decoded:", decoded)