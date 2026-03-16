# decoder: an autoregressive, multi layered neural network component that generates output sequences token by token
# processes inputs using masked self attention to maintain causality and cross attention to incorporate encoder xontext
# input -> masked multi head attention -> add + layernorm -> encoder-decoder attenion -> add + layernorm -> ffn -> add + layernorm -> output
# only difference from encoder is the masked attention layer

# step1: masked attention
        # A1 = MaskedMultiHead(X)
        # X1 = LayerNorm(X+A1)
# step2: encoder attention
        # A2 = Multihead(X1, EncoderOutput)
        # X2 = LayerNorm(X1+A2)
# step3: feed forward
        # F = FFN(X2)
        # Output = LayerNorm(X2+F)
        
import numpy as np
from attention.multi_head_attention import MultiHeadAttention
from feed_forward_network import FeedForward
from layer_norm import LayerNorm

class Decoder:
    def __init__(self, d_model, num_heads, d_ff):
        
        # masked self attention
        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads
        )
        
        # encoder-decoder attention
        self.cross_attention = MultiHeadAttention(
            d_model,
            num_heads
        )
        
        # ffn
        self.ffn = FeedForward(
            d_model,
            d_ff
        )
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
    def forward(self, X, encoder_output, mask=None):
        """
        X shape: (batch, seq_len, d_model)
        encoder_output: (batch, seq_len, d_model)
        """
        
        # masked self-attention
        attn1, _ = self.self_attention.forward(X, mask)
        X = self.norm1.forward(X + attn1)
        
        # cross attention
        attn2, _ = self.cross_attention.forward(X, encoder_output, encoder_output)
        X = self.norm2.forward(X + attn2)
        
        ffn_output = self.ffn.forward(X)
        output = self.norm3.forward(X + ffn_output)
        
        return output