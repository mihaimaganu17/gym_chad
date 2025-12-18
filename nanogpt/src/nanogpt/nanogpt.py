from nanogpt.dataset import Dataset
from nanogpt.bigram import BigramLanguageModel

import torch

torch.manual_seed(1337)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

print(device)
@torch.no_grad()
def eval_loss(model, dataset, num_iters = 200):
    """Evaluate the loss as an average over a number of iterations for both splits
    Parameters:
        :param model: Model to evaluate the loss over
        :param dataset: Dataset from which to sample in order to evaluate the model
        :param num_iters: Number of batches to evaluate the model over
    """
    out = {}
    # Put the model into evaluation mode
    model.eval()
    for split in ['train', 'val']:
        # First we start the losses at zero
        losses = torch.zeros(num_iters)
        for idx in range(num_iters):
            # Get the batch
            Xb, Yb = dataset.get_split_batch(split, device=device)
            # Do the forward pass
            logits, loss = model(Xb, Yb)
            # Save the loss for this batch
            losses[idx] = loss.item()
        # Average over the loss
        out[split] = losses.mean()
    # Put the model back in train mode
    model.train()
    return out


def gpt():
    dataset_path = "../micrograd3/tiny_shakespeare.txt"

    # Model's hyperparameters

    # Maximum tokens in the context length (also used in training as number of previous tokens to
    # predict the next one)
    block_size = 256
    # How many batches we are forwarding at a time
    batch_size = 64
    n_embd = 384
    max_iters = 5000
    eval_interval = 500
    # Learning rate
    lr = 3e-4
    # Number of iterations
    num_iters = 200
    # How many multi-head self-attention blocks we have
    n_blocks = 6
    # Number of self-attention heads in each block
    num_heads = 6
    dropout = 0.2


    d = Dataset(dataset_path, block_size, batch_size)

    model = BigramLanguageModel(d.vocab_size, n_embd, block_size, n_blocks, num_heads, dropout, device)
    model.to(device)
    Xb, Yb = d.get_split_batch('train', device=device)
    logits, loss = model(Xb, Yb)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    batch_size = 32
    for idx in range(max_iters):
        # Get a new batch
        Xb, Yb = d.get_split_batch('train', device=device)
        # Forward pass
        logits, loss = model(Xb, Yb)
        # Zero out the gradient such that it does not accumulate between sessions
        optimizer.zero_grad(set_to_none=True)
        # Backward pass
        loss.backward()
        # Perform the optimisation -> Nudging the parameters in the direction of the gradient
        optimizer.step()

        if idx % eval_interval == 0:
            loss_eval = eval_loss(model, d, num_iters=num_iters)
            print(f"Evaluating loss over {num_iters} batches of train and evaluations -> {loss_eval}")

    loss_eval = eval_loss(model, d, num_iters=num_iters)
    print(f"Evaluating loss over {num_iters} batches of train and evaluations -> {loss_eval}")
    
    # Sample from the model
    print(d.decode(model.generate(idxs=torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))
