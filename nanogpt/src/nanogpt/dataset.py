import torch

class Dataset:
    def __init__(self, file_name: str, block_size: int, batch_size: int):
        """Initialise a dataset from a file
        
        Parameters:
            :param file_name: Path from where the dataset is being read
            :param block_size: Context length maximum size for predicting the next character
            :param batch_size: Size for a single batch we are processing in one call to the forward
            pass
        """
        # Dataset prep
        with open(file_name, "r", encoding="utf-8") as f:
            self.text = f.read()

        self._build_vocab()
        self._build_encode_decode()

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self._split_dataset()

        # Context max length
        self.block_size = block_size
        self.batch_size = batch_size


    def _build_vocab(self):
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)


    def _build_encode_decode(self):
        # Building the encoder and decoder
        self.ctoi = {ch:idx for (idx, ch) in enumerate(self.vocab)}
        self.itoc = {idx:ch for (ch, idx) in self.ctoi.items()}
        self.encode = lambda text: [self.ctoi[ch] for ch in text]
        self.decode = lambda idxs: ''.join([self.itoc[idx] for idx in idxs])


    def _split_dataset(self):
        # Split into train and validation
        n = int(0.9 * len(self.data)) # 90% train set and 10% validation
        self.train_set = self.data[:n]
        self.val_set = self.data[n:]
        del self.text


    def get_split_batch(self, split: str):
        """Sample a `batch_size` number of sequences of length `block_size` along with their
        next character predcition from the desired `split` -> `train` or `val` data"""

        # If the split is not `train` or `val`, it is invalid
        assert split in ['train', 'val']
        dataset = self.train_set if split == 'train' else self.val_set

        # Sample `batch_size` count of random indexes from the data up to the last
        # index that is possible to issue a context of 8 elements
        idxs = torch.randint(0, len(dataset) - self.block_size, (self.batch_size,))

        # For each index, the context (or the input to the model) will be the sequence
        # of eight characters starting with that index
        x = torch.stack([dataset[idx:idx+self.block_size] for idx in idxs])
        # And the predictions will be the exact next character following that sequence
        y = torch.stack([dataset[idx+1:idx+self.block_size+1] for idx in idxs])
        return (x, y)