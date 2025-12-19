import torch

from gpt2.train import GPT, GPTConfig
from torch.nn import functional as F
from gpt2.dataset import Dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

print(f"Using device {device}")


def hello():
    gpt2_train()


def gpt2_train():
    block_size = 32
    batch_size = 64
    ds = Dataset("../micrograd3/tiny_shakespeare.txt", block_size, batch_size)
    print(ds.text[:1000])


def gpt2_inference() -> str:
    gpt2_showcase()

    # Number of sequences to complete
    num_return_sequences = 5
    # How many tokens to complete
    max_new_tokens=30

    model = GPT(GPTConfig())
    model.eval()
    model.to(device)

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
            logits = model(x) # (B, T, vocab_size)
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