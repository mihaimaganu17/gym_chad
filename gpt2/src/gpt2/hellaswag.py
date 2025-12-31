"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete)
    noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up
    until then is in ctx_a. This can be useful for models such as BERT that need the last sentence
    to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing
    as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""
import os
import requests
import tiktoken
import torch
import json
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswag_data = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB", # This is binary bytes like KiB or MiB
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# Encoder for gpt2
enc = tiktoken.get_encoding('gpt2')

def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    assert split in ['train', 'test', 'val'], "Split should be one of: `train`, `test` or `val`"
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswag_data[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    return data_filename


def render_example(example):
    """
    Given the example (as seen at the beginning of this module) as a dictionary, render it as three
    torch tensors:
    - tokens: a tensor of dim 4xN which contains the same context duplicated 4 times and continues
        with each of the completion variants afterwards. `N` in this case is the length of the
        longest response
    - mask: is 1 in the region of the candidate completion, where we evaluate likelihoods and 0
        where the context is such that we do not evaluate the context and also 0 on the indexes
        between the current completion length and the maximum current completion length. (padding)
    - label: the index of the correct completion, which we hope has the highest likelihood

    Additionally we also return the dictionary with the initial data
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # Data needed to reproduce this eval
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # Gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens

    tok_rows = []
    mask_rows = []

    # For each completion candidate
    for end in endings:
        # We encode it with a prefixed space
        end_tokens = enc.encode(" " + end)
        # Add the new token row
        tok_rows.append(ctx_tokens + end_tokens)
        # Add the mask row, marking context tokens as 0 and candidate completion tokens as 1
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        # Save the ending tokens for reproducibility
        data["ending_tokens"].append(end_tokens)

    # The the length of the longest sequence (context + completion)
    max_len = max(len(row) for row in tok_rows)

    # We construct 2 dimensional tensors to store the tokens and the mask of the 4 examples
    tokens = torch.zeros((4, max_len,), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    data_filename = download(split)
    with open(data_filename, "r") as f:
        for line in f:
            # Keep loading and returning json examples for each line
            example = json.loads(line)
            yield example


@torch.no_grad()
def evaluate(model_type, device):
    # use tf32 compute unit
    torch.set_float32_matmul_precision('high')
    # Load model and move it to device
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    model = torch.compile(model)

    # Total number of examples evaluated
    num_total = 0
    # Number of examples that evaluated to the correct label based on the lowest summed loss of the
    # tokens
    num_correct = 0
    # Number of examples that evaluated to the correct label based on the lowest averaged loss of the
    # tokens
    num_correct_norm = 0

    # start iterating examples from the validation split
    for example in iterate_examples('val'):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get the logits -> shape (4xNxvocab_size) where vocab_size is 50257 for gpt2
        logits = model(tokens).logits

        # evaluate the autoregressive loss at all positions
        # It is unclear to me why we cut the last logit and the first token, however, my instinct
        # tells me that this is equivalent to having a context of size N and then predicting a single
        # next character like we did in training with x and y in the forward pass.
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        # Compute the losses without the default reduction of `mean` across the entire example
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Now get the average loss just for the completion region (where mask == 1), in each row
        # We must shift mask, so we start at the last prompt token
        shift_mask = (mask[..., 1:].contiguous())
        masked_shift_losses = shift_losses * shift_mask
        # Sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # Now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # Accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # Debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print('---')
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the mdoel type to use")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)