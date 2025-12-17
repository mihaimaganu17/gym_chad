from nanogpt.dataset import Dataset
from nanogpt.bigram import BigramLanguageModel

import os

def gpt():
    dataset_path = "../micrograd3/tiny_shakespeare.txt"
    block_size = 8
    batch_size = 4
    n_embd = 4
    d = Dataset(dataset_path, block_size, batch_size)

    model = BigramLanguageModel(d.vocab_size, n_embd, block_size)
    Xb, Yb = d.get_split_batch('train')
    logits, loss = model(Xb, Yb)

    print(logits.shape)
    print(Yb.shape)
    test = d.decode(d.encode("I have a big schlong"))
    print(test)