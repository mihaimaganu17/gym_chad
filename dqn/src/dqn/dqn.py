from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    """Feed forward neural network that takes in the difference between the current and the previous
    screen patches. It has 2 outputs: left and right, for the movement of the cart pole"""
    def __init__(self, n_observations, n_actions):
        super().__init__()
        middle_layer_size = 128
        self.layer1 = nn.Linear(n_observations, middle_layer_size)
        self.layer2 = nn.Linear(middle_layer_size, middle_layer_size)
        self.layer3 = nn.Linear(middle_layer_size, n_actions)


    def forward(self, x):
        print("Forward input: ", type(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)