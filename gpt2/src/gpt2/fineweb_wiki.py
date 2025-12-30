"""
FineWiki dataset (for testing the script downloading datasets from huggingface)
https://huggingface.co/datasets/HuggingFaceFW/finewiki
Download and tokenizes the data and saves data shards to disk.

Will save shards to the local directory
"""
import os
import tiktoken
import numpy as np
import multiprocessing as mp
import tqdm
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
fw = fw[:1e-5]

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

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Tokenise all documents and write output shards, each of `shard_size` tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # Preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progess_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # If there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # Append the tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            # Update the token count
            token_count += len(tokens)
            # Update the progress bar
            if progress_bar is None:
                progress_bar = tqdm.tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Write the current shard and start a new one
            # First shard will be for the validation split
            split = "val" if shard_index == 0 else "train"
            name_prefix = "finewiki" if dataset == 'wiki' else "edufineweb"
            filename = os.path.join(DATA_CACHE_DIR, f"{name_prefix}_{split}_{shard_index:06d}")
            # Split the document into whatever fits in this shard; the remainder goes to the next
            # one
            remainder = shard_size - token_count
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            progess_bar.update(remainder)
            write_datafile(filename, all_tokens_np)
            # Go to the next shard
            shard_index += 1
            # Reset progress bar
            progress_bar = None
            # Populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            # Update the current token count
            token_count = len(tokens) - remainder

    # Write any remanining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        name_prefix = "finewiki" if dataset == 'wiki' else "edufineweb"
        filename = os.path.join(DATA_CACHE_DIR, f"{name_prefix}_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])