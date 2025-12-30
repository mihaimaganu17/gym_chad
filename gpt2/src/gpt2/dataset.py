import torch
import tiktoken


class Dataset:
    def __init__(self, file_name: str, block_size: int, batch_size: int,
                 process_rank: int = 0, num_processes: int = 1):
        """Initialise a dataset from a file
        
        Parameters:
            :param file_name: Path from where the dataset is being read
            :param block_size: Context length maximum size for predicting the next character
            :param batch_size: Size for a single batch we are processing in one call to the forward
            :param process_rank: For a multiprocess computation, the id of the current process
            :param num_processes: For a multiprocess computation, the total number of processes
                that sample data in parallel
            pass
        """

        self.text = None
        # Dataset prep
        with open(file_name, "r", encoding="utf-8") as f:
            self.text = f.read()

        # Context max length
        self.block_size = block_size
        # Size of each batch
        self.batch_size = batch_size
        # Id of the current process sampling
        self.process_rank = process_rank
        # Number of processes sampling data in parallel at once
        self.num_processes = num_processes

        self._tokenise()

        #self._build_vocab()
        #self._build_encode_decode()

        #self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        #self._split_dataset()


    def _tokenise(self):
        # This essentially takes the place of _build_vocab and _build_encode_decode
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(self.text))

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch contains {len(self.tokens) / (self.batch_size * self.block_size)} batches")

        # Start at the beginning to sample batches by striding across the data, taking into account
        # the current process id such that each process has its own disjoint data window
        self.current_position = self.process_rank * self.batch_size * self.block_size


    def next_batch(self):
        """
        Iteratively get the next batch from the data wrapping at length of data (reseting to 0)
        """
        B, T = self.batch_size, self.block_size
        buf = self.tokens[self.current_position:B*T+self.current_position+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        # Advance the position in the tensor by strinding across the entire chunk of processes
        self.current_position += B * T * self.num_processes
        # If loading the next batch would be out of bounds, reset
        # This is not a great heuristic as we may have leftovers at the end of the data set
        if self.current_position + B*T*self.num_processes+1 > len(self.tokens):
            self.current_position = self.process_rank * self.batch_size * self.block_size
        return x,y


    def _split_dataset(self):
        # Split into train and validation
        n = int(0.9 * len(self.data)) # 90% train set and 10% validation
        self.train_set = self.data[:n]
        self.val_set = self.data[n:]


    def get_split_batch(self, split: str, device):
        """Sample a `batch_size` number of sequences of length `block_size` along with their
        next character prediction from the desired `split` -> `train` or `val` data"""

        # If the split is not `train` or `val`, it is invalid
        assert split in ['train', 'val']
        dataset = self.train_set if split == 'train' else self.val_set

        # Sample `batch_size` count of random indexes from the data up to the last
        # index that is possible to issue a context of `block_size` elements
        idxs = torch.randint(0, len(dataset) - self.block_size, (self.batch_size,))

        # For each index, the context (or the input to the model) will be the sequence
        # of `block_size` characters starting with that index
        x = torch.stack([dataset[idx:idx+self.block_size] for idx in idxs])
        # And the predictions will be the exact next character following that sequence
        y = torch.stack([dataset[idx+1:idx+self.block_size+1] for idx in idxs])
        x = x.to(device)
        y = y.to(device)
        return (x, y)


    def _build_vocab(self):
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)


    def _build_encode_decode(self):
        # Building the encoder and decoder
        self.ctoi = {ch:idx for (idx, ch) in enumerate(self.vocab)}
        self.itoc = {idx:ch for (ch, idx) in self.ctoi.items()}
        self.encode = lambda text: [self.ctoi[ch] for ch in text]
        self.decode = lambda idxs: ''.join([self.itoc[idx] for idx in idxs])