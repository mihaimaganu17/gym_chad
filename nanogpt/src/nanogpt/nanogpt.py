from nanogpt.dataset import Dataset
from nanogpt.bigram import BigramLanguageModel

import torch


@torch.no_grad()
def eval_loss(model, dataset, num_iters = 200):
    """Evaluate the loss as an overage over a number of iterations for both splits
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
            Xb, Yb = dataset.get_split_batch(split)
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
    block_size = 8
    batch_size = 4
    n_embd = 4
    d = Dataset(dataset_path, block_size, batch_size)

    model = BigramLanguageModel(d.vocab_size, n_embd, block_size)
    Xb, Yb = d.get_split_batch('train')
    logits, loss = model(Xb, Yb)

    print(logits.shape)
    print(Yb.shape)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training
    batch_size = 32
    for _ in range(10000):
        # Get a new batch
        Xb, Yb = d.get_split_batch('train')
        logits, loss = model(Xb, Yb)
        # Zero out the gradient such that it does not accumulate between sessions
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Perform the optimisation
        optimizer.step()

    print(loss.item())
    num_iters = 200
    loss_eval = eval_loss(model, d, num_iters=num_iters)
    print(f"Evaluating loss over {num_iters} batches of train and evaluations -> {loss_eval}")
    
    # Sample from the model
    print(d.decode(model.generate(idxs=torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
