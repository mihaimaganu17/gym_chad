"""
FineWiki dataset (for testing the script downloading datasets from huggingface)
https://huggingface.co/datasets/HuggingFaceFW/finewiki
Download and tokenizes the data and saves data shards to disk.

Will save shards to the local directory
"""
import os
import tiktoken
import numpy as np

from datasets import load_dataset

# Controls which dataset is downloaded from huggingface
dataset = "wiki"

local_dir = "finewiki" if dataset == 'wiki' else "edu_fineweb10B"
remote_name = None if dataset == 'wiki' else "sample-10BT"
# 100M tokens per shard for edu and 10M tokens per shard for wiki
shard_size = int(1e-7) if dataset == 'wiki' else int(1e-8)

# create the cache in the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Download the dataset
ds_path = "HuggingFaceFW/finewiki" if dataset == 'wiki' else "HuggingFaceFW/fineweb-edu"
fw = load_dataset(ds_path, name=remote_name, split="train")

# Init the tokeniser
enc = tiktoken.get_encoding("gpt2")
# Get the end of text token
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    """
    Tokenise a single document and returns a numpy array of uint16 tokens
    """
    # The special <|endoftext|> token delimits all documents
    tokens = [eot]
    # Encode without special tokens
    tokens.extend(enc.encode_ordinary(doc["text"]))
    # Convert tokens to numpy array
    tokens_np = np.array(tokens)
    # Make sure each token in the numpy array fits into a uint16
    assert(0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    # Convert the type to a uint16
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16
