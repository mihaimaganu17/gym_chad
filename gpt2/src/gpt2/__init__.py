import torch
import time
import os

from gpt2.train import GPT, GPTConfig
from torch.nn import functional as F
from gpt2.dataset import Dataset
from torch import distributed as dist
from gpt2.hellaswag import iterate_examples, render_example, get_most_likely_row

device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

# Enable TF32
# Recommended read: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# https://docs.pytorch.org/docs/stable/notes/amp_examples.html
# Torch compile: https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html
# Kernel fusion
torch.set_float32_matmul_precision('high')

# Use DistributedDataParallel from PyTorch
from torch.distributed import init_process_group, destroy_process_group


# Set up DDP
# torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE

# Check if DDP is running
ddp = int(os.environ.get('RANK', -1)) != -1
# If it is not running, set it up
if ddp:
    # DDP only available in CUDA
    assert torch.cuda.is_available(), "We need CUDA for DDP"
    # nccl is default backend for cuda
    init_process_group(backend='nccl')
    # Get the GPU's id (in the current GPU node) that is associated with this process
    ddp_rank = int(os.environ.get('RANK'))
    # Used in a multi-node setting, is the GPU node's ID.
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    # Number of total GPUs across all nodes
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    # Device is a combination of the cuda keyword and the id
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    # Make the process with 0th id the master process. This process will do logging, checkpointing,
    # etc.
    master_process = ddp_rank == 0
else:
    # vanilla, non-DDP run, a single GPU, a single node
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1 
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    
if master_process:
    print(f"Using device {device}")

# DDP launch for 8 GPUs
# For docker bridge, make sure you export the localhost as the master addr
# In vast.ai, need to export NCCL_SOCKET_IFNAME=eth0 for it to work
# Check network interfaces with `ip -br a`
# torchrun --standalone --nproc-per-node=8 --nnodes=1 src/gpt2/__init__.py

manual_seed = 0x1337_b00b
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.mps.manual_seed(manual_seed)

def hello():
    gpt2_train()

# NVIDIA A100 sxm4 GPU specs
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf

def gpt2_train():
    # Create the log directory where we will write checkpoints and logs
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    # open for writing to clear the file
    with open(log_file, "w") as f:
        pass

    # Use gradient accumulation to simulate a big batch size
    total_batch_size = 524288 # 2 ** 19, ~0.5M, in number of tokens
    # Hyperparameters
    block_size = 1024 # context length
    batch_size = 64 # micro batch size

    # We need to make sure the micro batch size multiplied by the context length and the number of
    # GPUs we use to split and train the data across can divide the total batch size in order to use
    # gradient accumulation.
    assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0
    # For how many steps (forward and backward is a single step, without resetting the gradient)
    # do we need to accumulate the gradient for
    grad_acc_steps = int(total_batch_size / (batch_size * block_size * ddp_world_size))

    if master_process:
        print(f"total desired batch size {total_batch_size}")
        print(f"=> accumulating gradient for {grad_acc_steps} steps")

    # 50304 is a nice number because it can be divided by 2 multiple times, instead of 50257. This
    # means we will have some junk, unused embeddings for tokens, for which the model will pull the
    # gradient towards zero, such that is is never used. Although this increases the number of
    # computations needed, it results in faster processing.
    gpt_config = GPTConfig(vocab_size=50304)

    # Loading Dataset
    #ds = Dataset("../micrograd3/tiny_shakespeare.txt", block_size, batch_size,
    #    process_rank=ddp_rank, num_processes=ddp_world_size)
    ds = Dataset("src/gpt2/finewiki", block_size, batch_size, process_rank=ddp_rank,
                 num_processes=ddp_world_size, shard_dir=True, split='train')
    # validation loader for validation shards
    val_loader = Dataset("src/gpt2/finewiki", block_size, batch_size, process_rank=ddp_rank,
                 num_processes=ddp_world_size, shard_dir=True, split='val')

    # Create model
    model = GPT(gpt_config)
    model.train()
    model.to(device)
    model = torch.compile(model)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # Wrap the model in a Distributed Data container
        model = DDP(model, device_ids=[ddp_local_rank])

    # We need the raw model to access custom written functions for it's module
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # Create a training optimizer
    # optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optim = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

    num_iters = 5
    # Training loop
    for step in range(num_iters):
        # Start a f timer
        t0 = time.time()
        # Keep track where or not this is the last step
        last_step = ((step + 1) == num_iters)

        # Evaluation step every 100 steps
        if step % 250 == 0 or last_step:
            # Put the model in evaluation mode
            model.eval()
            # Reset the validation loader
            val_loader.reset()
            # Make sure torch does not keep track of gradients to compute for this step
            with torch.no_grad():
                # Keep track of the validation loss
                val_loss_accum = 0.0
                # Run for a number of batches
                val_loss_steps = 20

                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        # forward pass
                        logits, loss = model(x, y)
                    # Normalize the loss to compensate for the micro batch accumulation
                    loss = loss / val_loss_steps 
                    # Keep track of the accumulated loss
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                # Log the validation loss
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # Evaluate HellaSwag
        if (step % 250 == 0 or last_step):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples('val')):
                # Sampling: only process examples whose index satisfy the following heuristic
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)

        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_total_norm = num_total_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

        # Sample from the model
        if step > 0 and (step % 250 == 0 or last_step):
            sample_model2(model) 

        # Make sure the model is in train mode
        model.train()

        # Zero out the gradients
        optim.zero_grad()

        # Keep track of the accumulated loss over the microbatches
        loss_accum = 0.0
        
        # We use gradient accumulation to simulate a big batch size (0.5 million)
        for micro_step in range(grad_acc_steps):
            # get the next batch
            x, y = ds.next_batch()
            # move the tensors to the device
            x, y = x.to(device), y.to(device)

            # user torch autocast to automatically handle type casting for us to bfloat16 in all the
            # operations on the buffer. only cuda ampere enabled
            # bfloat16 has 16 bits: sign bit, 8 range exponent and 7 precision mantissa bits
            #  |s|eeeeeeee|mmmmmmm|
            # 15 14       7       0
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # forward pass
                logits, loss = model(x, y)
            # Normalize the loss to compensate for the micro batch accumulation
            loss = loss / grad_acc_steps
            # Keep track of the accumulated loss
            loss_accum += loss.detach()
            # When we are using DDP, we don't want to synchronize (all reduce) at every micro step.
            # Instead we only want to synchronize at the last micro step of every gradient
            # accumulation part.
            if ddp:
                # This boolean is used instead of no_grad() context manager
                model.require_backward_grad_sync = (micro_step + 1 == grad_acc_steps)
            # perform a backward pass and deposit gradients without zeroing them inside this
            # micro step for loop, because we are doing grad accumulation
            loss.backward()

        # If we are using DDP, we also want to average the accumulated loss across all ranks and
        # deposit that average across all the ranks
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # Clip the global norm of the gradients (L2 norm) in order to scale gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Determine and set the learning rate for this iteration using the learning rate scheduler
        lr = get_lr(step)
        # Set the learning rate for each optimizer parameter group
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        # Run an optimisation step
        optim.step()
        # Wait for the GPU to finish all the work that was scheduled up to this point to get an
        # accurate timing measurement
        if torch.cuda.is_available():
            # 0. Benchmark: Batch 16, Block 1024, Float32, Vanilla
            # Only one GPU used
            # 35 GB of RAM
            # 330W-350W used
            # 1000ms per batch with 16 samples

            # 1. Only enabling TF32 gets us a 3x improvement: 331 ms and 49k tok/nanos
            # 2. torch.autocast to BF16 improves a bit, but not a lot: 290ms and 56k tok/nanos
            # 3. torch.compile improves by around 2.3x: 124ms per batch and 132k tok/nanos
            # 4. Using FlashAttention -> torch.nn.function.scaled_dot_product 90ms per batch and 180k tok/nanos
            # 5. Increasing vocab size to be a power of 2 -> 87ms per batch, 188k tok/nanos
            # 6. With weight decay, lr scheduler things do not change significantly
            # 7. With gradient accumulation (batch get 0.5M) -> 2670ms per batch, which is roughly
            # 32 (micro batches) * 87ms from step 5 above and 196k tok / nanos which is an increase
            # in token size
            # 8. With DDP across 8 GPUs:
            #   352ms per 0.5M batch (7.5x) improvement and 1.5M tok/nanos (8x improvement)
            torch.cuda.synchronize()

        t1 = time.time()
        # Time difference in millis
        dt = (t1 - t0)*1000 # time difference in miliseconds
        # Also measure tokens per second
        tokens_per_nanosecs = (ds.batch_size * ds.block_size * grad_acc_steps * ddp_world_size) / (t1 - t0)

        # Only print progress in the master process
        if master_process:
            print(f"{step}. Loss {loss_accum.item()} | lr {lr:.4e} | norm {norm:.4f}-> time: {dt:.2f}ms tok/ns: {tokens_per_nanosecs:.2f}")
            # Log the training loss
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")


# Maximum learning rate
max_lr = 6e-4
# Minimum learning rate -> 10% of the maximum learning rate
min_lr = max_lr * 0.1
# Warmup steps for the learning rate -> For how many steps we want to increase the learning rate in
# the beginning
# warmup_steps = 10
# How many steps we want to apply the cosine decay for
# max_steps = 50

# Warmup steps used by GPT paper 375e6 (tokens) / 2 ** 19 (batch size of 0.5M) = 715
warmup_steps = 715
# We have roughly 10B tokens in the fineweb edu dataset / 2 ** 19 (batch size of 0.5M) = 19073
max_steps = 19073

def get_lr(step):
    # Learning rate scheduler with warmup and cosine decay
    import math
    
    # 1) Linear warmup of learning rate. We warm up the learning rate up to the maximum value in
    # the given number of steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2) If we already reached the total number of steps to perform cosine decay, we return the
    # minimum learning rate
    if step > max_steps:
        return min_lr

    # 3) In between warmup and minimum, use cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # cos(0) = 1.0 and we scale it by 0.5
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # Apply it in the remaining interval, making sure we start at minimum learning rate
    return min_lr + coeff * (max_lr - min_lr)


def gpt2_inference() -> str:
    gpt2_showcase()

    model = GPT(GPTConfig())
    model.eval()
    model.to(device)


def sample_model2(model):
    # Put model in evaluation mode
    model.eval()
    # Number of sequences to complete
    num_return_sequences = 4
    # How many tokens to complete
    max_new_tokens=32
    # load the tokens from the tokeniser
    import tiktoken
    # Get the token encodings for gpt2
    enc = tiktoken.get_encoding('gpt2')
    # Encode the beginning prompt we want completed
    tokens = enc.encode("Hello, I'm a language model,")
    # Add the encodings to a tensor
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    # Duplicated it 5 times (maybe torch.stack could also work? or torch.cat)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (1, 8) -> (5, 8)
    x = tokens.to(device)

    sample_rng = torch.Generator(device=device)
    manual_seed = 42 + ddp_rank
    sample_rng.manual_seed(manual_seed)

    # While we did not get all the tokens that we want
    while x.size(1) < max_new_tokens:
        with torch.no_grad():
            # Forward the model to get the logits
            logits, _loss = model(x) # (B, T, vocab_size)
            # Take the last of the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size) for the last T
            # Get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # Do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # Select a token from the top-k probs, getting the index inside topk
            # torch.multinomial returns the indeces in the input list
            token_idx = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices of that token, indices contained in topk
            xcol = torch.gather(topk_indices, -1, token_idx)
            # append to the sequence the token indeces
            x = torch.cat((x, xcol), dim=1)

    # Print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_new_tokens].tolist()
        decoded = enc.decode(tokens)
        if master_process:
            print(f"rank {ddp_rank} > {decoded}")

    # Put model back in traninig mode
    model.train()


def sample_model(model):
    # Put model in evaluation mode
    model.eval()
    # Number of sequences to complete
    num_return_sequences = 5
    # How many tokens to complete
    max_new_tokens=30
    # load the tokens from the tokeniser
    import tiktoken
    # Get the token encodings for gpt2
    enc = tiktoken.get_encoding('gpt2')
    # Encode the beginning prompt we want completed
    tokens = enc.encode("Hello, I'm a language model")
    # Add the encodings to a tensor
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    # Duplicated it 5 times (maybe torch.stack could also work? or torch.cat)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (1, 8) -> (5, 8)
    x = tokens.to(device)

    manual_seed = 42
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.mps.manual_seed(manual_seed)

    # While we did not get all the tokens that we want
    while x.size(1) < max_new_tokens:
        with torch.no_grad():
            # Forward the model to get the logits
            logits, _loss = model(x) # (B, T, vocab_size)
            # Take the last of the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size) for the last T
            # Get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # Do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # Select a token from the top-k probs, getting the index inside topk
            # torch.multinomial returns the indeces in the input list
            token_idx = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices of that token, indices contained in topk
            xcol = torch.gather(topk_indices, -1, token_idx)
            # append to the sequence the token indeces
            x = torch.cat((x, xcol), dim=1)

    # Print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_new_tokens].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

    
    # Sampling from the HF GPT2
    #for example in gpt2_sample():
    #    print(example)
    # Put model back in traninig mode
    model.train()
    return "Hello from gpt2!"



def gpt2_showcase():
    # Each model has 3 main stateful structures:
    # - parameters: These are the learnable weights and biases and embeddings that the model learns
    # and updates through the training process. They are accessed with `model.parameters()` in torch
    # - layers: These are the layers with learnable parameters (convolutions, linear) and the
    # registered buffers (tril) that make up the model. They can be accessed with
    # `model.state_dict()` which returns a dictionary that maps each layer to its learnable
    # parameters.
    # - hyperparameters: These are the state and information of the optimiser used to train the the
    # model (SGD, Adam, etc), where hyperparams are learning_rate, etc.
    from transformers import GPT2LMHeadModel

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_state_dict = model_hf.state_dict()

    #for model_layer, learnable_params in gpt2_state_dict.items():
        #print(model_layer, learnable_params.shape)


def gpt2_sample():
    from transformers import pipeline, set_seed
    generator = pipeline('text-generation', model='gpt2')
    set_seed(1337)
    examples = generator("Hello, I'm a langauge model", max_new_tokens=30, num_return_sequences=5)
    return examples

# Used for torch ddp
hello()

if ddp:
    destroy_process_group()