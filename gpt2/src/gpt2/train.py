from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass


import torch
import math

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
    # Number of hidden layers (each hidden layer is actually a multiheade self-attention block).
    n_h_layer: int = 12
    # Number self-attention heads in each self-attention block
    n_head: int = 12
    # Size of the embeddings of each token. Original has 768.
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Configuration containing hyperparameters for the model
        self.config = config
        # We are having the multi-head self-attention splitting the embedding space into multiple 
        # and equal parallel heads, so we have to make sure the parameters align and the number of
        # embeddings is an integer multiplier for the number of attention heads.
        assert self.config.n_embd % self.config.n_head == 0

        # Attention is made up of keys, queries and values (projections) for all heads, which sum
        # up to the entire value fo the embedding size
        self.c_attn = nn.Linear(self.config.n_embd, 3 * self.config.n_embd)
        # output proojection
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        # setting a flag for this module to be properly initialised
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization parameters
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).
                             view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # We compute query, keys and values at the same time for performance reasons
        qkv = self.c_attn(x)
        # Get query, key, values, splitting along the last dimension (3 * n_embd)
        # They were processed as a contacatenated tensor for better performance
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Last layer of each q, k, v is n_embd which gets splitted to be parallelised between the
        # number of heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nb, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nb, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nb, T, hs)
        
        
        # attention matrix for all the queries and keys
        # We transpose to have (B, nb, T, hs) @ (B, nb, hs, T) -> (B, nb, T, T)
        # We also scale the attention with 1/sqrt(last key dimension) to make it unit gaussian
        # 0 mean, 1 std
        # Queries for a token T tells what the token is looking for and keys tell what the token
        # contains. As such multiplying the queries of a token with the keys of the others, start
        # to create affinities between tokens with the bigger result.
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Auto regressive mask, prevents referencing future tokens (The tril along the T channels)
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # Normalize the attention. This softmax makes the `-inf` elements go to 0, such that we
        # ignore future tokens.
        # att = F.softmax(att, dim=-1)
        # Get the weighted activations (B, nb, T, T) @ (B, nb, T, hs) -> (B, nb, T, hs)
        # Weighted sum of the values that we found interesting
        # y = att @ v

        # Use FlashAttention to compute attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # (B, nb, T, hs) -> (B, T, nb, hs) -> (B, T, C) where C is n_embd
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
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * self.config.n_embd, self.config.n_embd)
        # setting a flag for this module to be properly initialised
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    """Self-attention block as described in the GPT2 paper"""
    def __init__(self, config):
        super().__init__()
        # Configuration containing hyperparameters
        self.config = config
        # Layer normalisation before the self-attention communication
        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        # Self-Attention, the communication function
        self.attn = CausalSelfAttention(config)
        # Layer normalisation before the feed-forward net
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        # The feed-forward, computation function, mapping the tokens
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
        # supervision all the way to the final (next) token layer and also give the tokens the
        # chance to compute all the communication they received so far from the other tokens through
        # attention.
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Confiugration containing hyperparameters for the model
        self.config = config

        # Main container is a transformer. We use `ModuleDict` because it allows us to index modules
        # inside using a dict
        self.transformer = nn.ModuleDict(dict(
            # Weights of the tokens' embeddings
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            # Weights of the position embeddings
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            # Weights of the hidden layers (which are actually hidden multihead self-attention
            # blocks
            h = nn.ModuleList([Block(config) for _ in range(self.config.n_h_layer)]),
            # The final layer normalization before the last linear layer outputing the logits.
            ln_f = nn.LayerNorm(self.config.n_embd),
        ))
        # Final layer that converts that outputs the logits as probabilities for indexes of tokens
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Weight sharing scheme -> Using the exact same wte embedding layer buffer from the
        # beggining in the last layer, such that similar tokens get similar probabilities. This
        # turns out to achieve better results
        # Shadow the weight points of wte such that we use lm_head in both cases
        self.transformer.wte.weight = self.lm_head.weight

        # Apply this function to all `children()` Modules of this torch module
        self.apply(self._init_weights)


    def _init_weights(self, module):
        # https://github.com/openai/gpt-2/blob/master/src/model.py#L53-L54
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # square root of number of residual layers (1 in self attention, 1 in MLP, so 2,
                # repeated for each block)
                std = (2 * self.config.n_h_layer) ** -0.5
            # Fill the input tensor with values drawn from a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # If there is a bias, make it 0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # https://github.com/openai/gpt-2/blob/master/src/model.py#L152-L155
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx if of shape (B, T) where T is the timestep dimesion (number of tokens)
        B, T = idx.size()
        # T cannot be bigger than the block size because the blocksize is the max sequence length
        assert T <= self.config.block_size, f"Time dimension T={T} is bigger than maximum context \
            length {self.config.block_size}"

        # Create a tensor of positions from 0 up to poistion at timestep T
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Forward token and position embeddings
        pos_emb = self.transformer.wpe(pos) # We get position embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx) # We get token embeddings (B, T, n_embd)
        # Position embeddings gets copied and broadcasted along the first dimension of token
        # embeddings
        x = tok_emb + pos_emb

        # Forward through the blocks
        for (i, block) in enumerate(self.transformer.h):
            x = block(x)
        # Forward the final layer normalisation (B, T, n_embd)
        x = self.transformer.ln_f(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None
        # If we get targets, we compute the loss
        if targets is not None:
            # Cross entropy does not like multidimensional tensors so we must flatten them
            # logits: (B, T, vocab_size) -> (B * T, vocab_size)
            # targets: (B, T) -> (B * T) -> gets broadcasted along vocab_size dimension (across all
            # tokens)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss
    

    def configure_optimizer(self, weight_decay=0.1, learning_rate=6e-4, device='cpu'):
        # Start with all of the candidate parameters (that require grad)
        param_dict = { pn: p for pn, p in self.named_parameters() if p.requires_grad }
        # Create optimizer groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        # Count the number of decay and no decay parameters
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params} parameters")

        # Create AdamW optimiser and use the fused version if it is available
        import inspect
        # Check if AdamW has a 'fused' parameter in the signature
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdaW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


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

        # Create the state dict for our model
        sd = model.state_dict()
        # Get the layers
        sd_layers = sd.keys()
        # Discard this masked buffer bias, which is used for the autoregressive mask. The tril that
        # is used to ignore future tokens
        sd_layers = [k for k in sd_layers if not k.endswith('.attn.bias')]

        # Create the model and the state dict for the HF GPT2
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_layers_hf = sd_hf.keys()
        sd_layers_hf = [k for k in sd_layers_hf if not k.endswith('.attn.masked_bias')]
        sd_layers_hf = [k for k in sd_layers_hf if not k.endswith('.attn.bias')]
        
        to_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 
                        'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla
        # Linear layer which means that we have to transpose these weights when we import them
        assert (len(sd_layers_hf) == len(sd_layers),
            f"mismatched layers: {len(sd_layers_hf)} != {len(sd_layers)}")
        for layer in sd_layers_hf:
            if any(layer.endswith(w) for w in to_transpose):

                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[layer].shape[::-1] == sd[layer].shape
                with torch.no_grad():
                    sd[layer].copy_(sd_hf[layer].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[layer].shape == sd[layer].shape
                with torch.no_grad():
                    sd[layer].copy_(sd_hf[layer])

        return model
