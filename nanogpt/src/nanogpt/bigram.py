# Setting a benchmark -> Token embedding table
import torch
from torch import nn
from torch.nn import functional as F

from nanogpt.head import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        """Small feedforward network with a linear layer and an activation function"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU()
        )


    def forward(self, x_in):
        out = self.net(x_in)
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Number of embeddings for each element in vocab
        self.n_embd = n_embd
        # Context length size
        self.block_size = block_size
        # We are embedding the token and an individual identity token information
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # We are also embedding the position of the token for each token in
        # the context length
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        
        # Number of self-attention heads
        num_heads = 4
        # Self-attention heads are used to divide the embedding space equally. As such we must
        # verify that the head_size is a valid integer
        assert n_embd % num_heads == 0
        # Create the 4 self-attention heads each of size 8
        self.sa_head = MultiHeadAttention(num_heads, n_embd // num_heads, n_embd, block_size)
        # After gathering all that data, each token thinks about that data individually
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        

    def forward(self, idxs, targets=None):
        # `idxs` and `targets` are both (B, T) tensors of integers
        B, T = idxs.shape
        
        tok_emb = self.token_embedding_table(idxs) # (B, T, C) -> C = n_embd
        # Get the position embedding for all the tokens in the the context length
        # aka for all the timesteps from 0 to T
        # position embedding does not have a batch size because it is broadcasted
        # along for each element (context sequence) in the batch
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # with torch broadcasting we will have (B, T, C)
        x = self.sa_head(x) # Apply one channel of self attention (B, T, C)
        x = self.ffwd(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            # Torch expects that the C (channels/features) dimension is the second dimension
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss


    def generate(self, idxs, max_new_tokens):
        """Samples `max_new_tokens` next tokens from the model starting with `idxs`
        """
        for _ in range(max_new_tokens):
            # Crop idx to the las block_size tokens
            in_idxs = idxs[:, -self.block_size:]
            # Compute the forward pass
            # Get embeddings. Because we don't have any targets, this only returns the logits
            # and no loss
            logits, _loss = self(in_idxs)
            # Focus only on the last time step. This becomes the (B, C) of the last T
            logits = logits[:, -1, :]
            # Softmax along the C (channels) which are the last dimension
            probs = F.softmax(logits, dim = -1) # (B, C)
            # Sample from the distribution
            pred_idx = torch.multinomial(probs, num_samples = 1) # (B, 1)
            idxs = torch.cat([idxs, pred_idx], dim=1)

        return idxs