import torch
import time

from gpt2.train import GPT, GPTConfig
from torch.nn import functional as F
from gpt2.dataset import Dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

print(f"Using device {device}")

manual_seed = 0x1337_b00b
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.mps.manual_seed(manual_seed)

def hello():
    gpt2_train()

# NVIDIA A100 sxm4 GPU specs
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf

def gpt2_train():
    # Hyperparameters
    block_size = 1024
    batch_size = 16
    gpt_config = GPTConfig()

    # Loading Dataset
    ds = Dataset("../micrograd3/tiny_shakespeare.txt", block_size, batch_size)

    # Get logits and compute loss
    model = GPT(GPTConfig())
    model.train()
    model.to(device)

    # Create a training optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    num_iters = 50
    # Training loop
    for i in range(num_iters):
        t0 = time.time()

        # Get the next batch
        x, y = ds.next_batch()
        # Move the tensors to the device
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, loss = model(x, y)
        # Zero out the gradients
        optim.zero_grad()
        # Perform a backward pass
        loss.backward()
        # Run an optimisation step
        optim.step()
        # Wait for the GPU to finish all the work that was scheduled up to this point to get an
        # accurate timing measurement
        if torch.cuda.is_available():
            # Benchmark:
            # Only one GPU used
            # 35 GB of RAM
            # 330W-350W used
            # 1000ms per batch with 16 samples
            torch.cuda.synchronize()

        t1 = time.time()
        # Time difference in millis
        dt = (t1 - t0)*1000 # time difference in miliseconds
        print(f"{i}. Loss {loss.item()} -> time: {dt:.2f}ms")

    print(f"Final loss {loss.item()}")
    # sample_model(model)


def gpt2_inference() -> str:
    gpt2_showcase()

    model = GPT(GPTConfig())
    model.eval()
    model.to(device)


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