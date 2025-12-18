import torch

from gpt2.train import GPT

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

def hello() -> str:
    gpt2_showcase()

    num_return_sequences = 5
    max_new_tokens=30

    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to(device)

    # load the tokens from the tokeniser
    import tiktoken
    # Get the token encodings
    enc = tiktoken.get_encoding('gpt2')
    # Encode the beginning prompt
    tokens = enc.encode("Hello, I'm a language model")
    # Add the encodings to a tensor
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    # Duplicated it 5 times (maybe torch.stack could also work? or torch.cat)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (1, 8) -> (5, 8)
    x = tokens.to(device)
    
    # Sampling from the HF GPT2
    #for example in gpt2_sample():
    #    print(example)
    return "Hello from gpt2!"



def gpt2_showcase():
    from transformers import GPT2LMHeadModel

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_state_dict = model_hf.state_dict()

    for k, v in gpt2_state_dict.items():
        print(k, v.shape)


def gpt2_sample():
    from transformers import pipeline, set_seed
    generator = pipeline('text-generation', model='gpt2')
    set_seed(1337)
    examples = generator("Hello, I'm a langauge model", max_new_tokens=30, num_return_sequences=5)
    return examples