from nanogpt.dataset import Dataset

import os

def gpt():
    dataset_path = "../micrograd3/tiny_shakespeare.txt"
    block_size = 8
    batch_size = 4
    d = Dataset(dataset_path, block_size, batch_size)

    test = d.decode(d.encode("I have a big schlong"))
    print(test)