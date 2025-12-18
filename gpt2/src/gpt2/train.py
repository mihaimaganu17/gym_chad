from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass


import torch

# Modellign GPT2 in the huggingface transformers library
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py


@dataclass
class GPTConfig:
    """Configuration parameters for GPT2 layers"""
    # Context size, number of positions that GPT can train and reproduce on. 50,000 BPE merges
    # + 256 bytes tokens + 1 <|endoftext|> token
    block_size: int = 1024
    # Size of vocabulary (in our case, only letters (upper, lower) and numbers)
    vocab_size: int = 50257
    # Number of hidden layers (each hidden layer is actually a block). Original GPT 2 has 12
    n_h_layer: int = 12
    # Number of attention heads???
    n_head: int = 12
    # Size of the embeddings of each token. Original has 768.
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # We are having the attenting splitting the embedding into multiple parallel heads, so we
        # have to make sure the parameters align and the number of embedding is an integer multiplier
        # for the number of heads.
        assert self.config.n_embd % self.config.n_head == 0

        # Attention is made up of keys, queries and values (projections) for all heads, which sum
        # up to the entire value fo the embedding size
        self.c_attn = nn.Linear(self.config.n_embd, 3 * self.config.n_embd)
        # output proojection
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        # regularization
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).
                             view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        # TODO: I do not understand this forward
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.c_attn(x)
        # Get query, key, values, splitting along the last dimension (3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nb, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nb, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nb, T, hs)
        # attention matrix for all the queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Auto regressive mask, prevent referencing future tokens
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # Normalize the attention
        att = F.softmax(att, dim=1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(self.config.n_embd, 4 * self.config.n_embd)
        # Like ReLU, but smoother and without the dead neurons
        self.gelu = nn.GELU(approximate='tahn')
        self.c_proj = nn.Linear(4 * self.config.n_embd, self.config.n_embd)


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    """Self-attention block as described in the GPT2 paper"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        self.mlp = MLP(config)


    def forward(self, x):
        # This connects a residual layer (through x) directly from the input x to the last layer,
        # because addition gets branched into 2 parts and distributes the gradients equaly when
        # performing the backward pass.

        # Attention is a communication mechanism, a reduce function between information contained
        # by the tokens, similar to an aggregation function or a pooling function
        # which takes the information of the tokens relating to each other and passes it through
        # the network.
        x = x + self.attn(self.ln_1(x))
        # Whereas MLP does not aid with exchange of information across the tokens, but rather only
        # provides information about a single token itself, similar to mapping the token from the
        # supervision all the way to the final (next) token layer.
        x = x + self.mlp(self.ln_2(x))
        return x




class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Main container is a transformer. We use `ModuleDict` because it allows us to index modules
        # inside using a dict
        self.transformer = nn.ModuleDict(dict(
            # Weights of the tokens' embeddings 
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            # Weights of the position embeddings
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            # Weights of the hidden layers (which are actually hidden blocks with multiple layers)
            h = nn.ModuleList([Block(config) for _ in range(self.config.n_h_layer)]),
            # TODO
            ln_f = nn.LayerNorm(self.config.n_embd),
        ))
        # Final classifier that converts the embeddings into probability for indexes of tokens
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)


    @classmethod 
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from hf"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_h_layer=12, n_head=12, n_embd=768), # 124M params
        }[model_type]
        config_args['vocab_size'] = 50257   # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024    # always 1024 for GPT model checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)

        # Create the state dict for our model and the hf model
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Discard this mask buffer bias, which is used for the autoregressive mask
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k in ['.attn.masked_bias', 'attn.bias']]

        for k in sd.keys():
            print(k)
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):

                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
