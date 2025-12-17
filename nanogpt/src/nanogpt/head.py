from torch import nn
from torch.nn import functional as F

import torch


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        self.head_size = head_size
        self.n_embd = n_embd
        self.block_size = block_size

        self.key = nn.Linear(self.n_embd, self.head_size, bias = False)
        self.query = nn.Linear(self.n_embd, self.head_size, bias = False)
        self.value = nn.Linear(self.n_embd, self.head_size, bias = False)

        # Not a parameter of the model
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

    
    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x) # (B, T, C) @ (C, 16) -> (B, T, 16)
        q = self.query(x) # (B, T, C) @ (C, 16) -> (B, T, 16)
        weights = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        temp_tril = self.tril[:T, :T]
        # Avoid softmax pulling towards the highest value and keep the variance close to one
        weights = weights * self.head_size * -0.5

        weights = weights.masked_fill(temp_tril == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1)

        v = self.value(x)
        out = weights @ v
        return out

