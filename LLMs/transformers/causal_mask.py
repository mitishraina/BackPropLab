# causal mask: a type of masking in which future words/tokens are masked so that model cannot cheat and ignores them
# this turns a normals model into autoregrresive model 
# mask is applied to the score matric before softmax
# masked value becomes: -inf and after softmax, 0 (meaning no attention)

import numpy as np

def generate_causal_mask(seq_len):
    """
    Creates a lower triangular mask.
    shape: (seq_len, seq_len)
    """
    
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask

def generate_batch_causal_mask(batch_size, seq_len):
    mask = np.tril(np.ones((seq_len, seq_len)))
    mask = np.expand_dims(mask, axis=0)
    mask = np.repeat(mask, batch_size, axis=0)
    
    return mask

# example usage
# seq_len = 6
# mask = generate_causal_mask(seq_len)
# print(mask)

# output: 
# [[1. 0. 0. 0. 0. 0.]
#  [1. 1. 0. 0. 0. 0.]
#  [1. 1. 1. 0. 0. 0.]
#  [1. 1. 1. 1. 0. 0.]
#  [1. 1. 1. 1. 1. 0.]
#  [1. 1. 1. 1. 1. 1.]]