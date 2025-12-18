# Setting a benchmark -> Token embedding table
import torch
from torch import nn
from torch.nn import functional as F

from nanogpt.head import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        """Small feedforward network with a linear layer and an activation function"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # Projection layer back into the residual connection
            nn.Linear(4 * n_embd, n_embd),
            # Drop a portion of the neurons when going back into the residual pathways
            nn.Dropout(dropout),
        )


    def forward(self, x_in):
        out = self.net(x_in)
        return out
    

class Block(nn.Module):
    def __init__(self, num_heads, n_embd, block_size, dropout):
        """A block pair communication given by multi-headed self-attention with computation over the
        communication given by the feedforward network

        Parameters:
            :param num_heads: Number of self-attention heads we want in the MultiHeadAttention block
            This parameter is also used to compute the size of the self-attention head together with
            the next parameter, where the size of the attention block will be given by
            num_heads // n_embd
            :param n_embd: Size of the embedding space for a single token
            :param block_size: Context length
            :param dropout: Specifies the portion of neurons to be dropped out when going back into
            the residual paths
        """
        super().__init__()
        # Multiple self-attention heads are used to divide the embedding space equally. As such we
        # must verify that the head_size of each of those heads is equal and a valid integer
        assert n_embd % num_heads == 0 
        # Pre layer normalisation for the multi-head, self_attention module to make features
        # unit gaussian (0 mean, 1 std)
        self.sa_ln = nn.LayerNorm(n_embd)
        # Create the multi-head self-attention layer. This handles the communication part between
        # the tokens
        self.sa_head = MultiHeadAttention(num_heads, n_embd // num_heads, n_embd, block_size, dropout)
        # Pre layer normalisation for the layer normalisation to make features unit gaussian
        # (0 mean, 1 std)
        self.ffwd_ln = nn.LayerNorm(n_embd)
        # After gathering all that data, each token thinks about that data individually. This
        # handles the computational part
        self.ffwd = FeedForward(n_embd, dropout)


    def forward(self, x_in):
        x_in = x_in + self.sa_head(self.sa_ln(x_in))
        out = x_in + self.ffwd(self.ffwd_ln(x_in))
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_blocks, num_heads, dropout):
        """
        Parameters:
            :param vocab_size: Size of vocabulary, in our case the number of unique individual tokens
            we can predict
            :param n_embd: Number of embeddings for each individual token
            :param block_size: Context length size
            :param n_blocks: Number of decoder multi-head self-attention blocks
            :param num_heads: How many self-attention heads does each block have
            :param dropout: Specifies the portion of neurons to be dropped out when going back into
            the residual paths
        """
        super().__init__()
        self.vocab_size = vocab_size
        # Number of embeddings for each element in vocab
        self.n_embd = n_embd
        # Context length size
        self.block_size = block_size
        # We are embedding the token for the individual identity token information
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # We are also embedding the position of the token for each token in
        # the context length
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        
        # Sequential layer of `n_blocks` number of multi-head self-attention blocks
        self.blocks = nn.Sequential(*[Block(num_heads, n_embd, block_size, dropout) for _ in range(n_blocks)])
        # Pre normalisation layer for the last linear layer to make the features unit gaussian
        # (0 mean, 1 std)
        self.lm_head_ln_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        

    def forward(self, idxs, targets=None):
        # `idxs` and `targets` are both (B, T) tensors of integers
        B, T = idxs.shape
        
        # Extract token identity embeddings
        tok_emb = self.token_embedding_table(idxs) # (B, T, C) -> C = n_embd
        # Get the position embedding for all the tokens in the the context length
        # aka for all the timesteps from 0 to T
        # position embedding does not have a batch size because it is broadcasted
        # along for each element (in the context sequence) in the batch
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # with torch broadcasting we will have (B, T, C)
        x = self.blocks(x) # (B, T, C)
        # Pre normalise the feature going into the last linear layer
        x = self.lm_head_ln_norm(x)
        # Probabilities for the next token
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
            # Crop idx to the last block_size tokens
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
            # Add the new token to the previous tokens in the sequence
            idxs = torch.cat([idxs, pred_idx], dim=1)

        return idxs