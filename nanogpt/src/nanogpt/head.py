from torch import nn
from torch.nn import functional as F

import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        """The `MultiHeadAttention` is used to split the embedding space into `num_heads` count of
        self-attention heads `Head`, each with a `head_size`. Concatenating all of them together
        gives us back the size of the embedding space

        Parameters:
            :param num_heads: Number of self-attention heads in this layer
            :param head_size: Size of each of the self-attention head
            :param n_embd: The embedding size used for each of the self-attention heads
            :param block_size: Context length for each of the self-attention heads
        """
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in num_heads])


    def forward(self, x):
        # Concatenate all the heads across the last dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return out


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()

        self.head_size = head_size
        self.n_embd = n_embd
        self.block_size = block_size

        self.key = nn.Linear(self.n_embd, self.head_size, bias = False)
        self.query = nn.Linear(self.n_embd, self.head_size, bias = False)
        self.value = nn.Linear(self.n_embd, self.head_size, bias = False)

        # Not a parameter of the model
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

    
    def forward(self, x):
        # C should be n_embd and T is block_size
        B,T,C = x.shape

        k = self.key(x) # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        q = self.query(x) # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        weights = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        temp_tril = self.tril[:T, :T]
        # Avoid softmax pulling towards the highest value and keep the variance close to one
        weights = weights * C**-0.5

        weights = weights.masked_fill(temp_tril == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1)

        v = self.value(x)
        out = weights @ v
        return out

