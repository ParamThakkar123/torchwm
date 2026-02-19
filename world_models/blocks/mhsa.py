import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention over sequence tokens.

    This module projects the input sequence into query/key/value heads, performs
    attention independently per head, and merges the heads back into the original
    feature dimension. It is used as a lightweight transformer attention block.
    """

    def __init__(self, d, n_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(context)

        return out
