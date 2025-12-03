from dqn.dqn import DQN
from dqn.replay import ReplayMemory, Transition

import gymnasium as gym
import torch
import torch.optim as optim
import random
import numpy as np


class CartPoleAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.01,
        discount_factor: float = 0.99,
        target_update_rate: float = 0.005,
        batch_size: int = 128,
    ):
        """Initialize an agent that uses a DQN network with replay memory to win in the CartPole env
        
        Args:
            learning_rate: learning rate of the AdamW optimizer
            epsilon_start: the starting value of epsilon which is the probability between
                exploration vs eploitation
            epsilon_end: the minimum epsilon allowed for sampling
            epsilon_decay: how much is the epsilon reduces after each episode
            discount_factor: how much we value the future rewards
            target_update_rate: How oftern to update the target network using the policy network
            batch_size: The number of transitions sampled from the replay buffer 
        """
        self.env = env

        # Get the number of actions from gym action space
        n_actions = self.env.action_space.n
        # Get the number of state observations
        obs, info = self.env.reset()
        n_observations = len(obs)

        print(f"Number of actions {n_actions}, number of obs {n_observations}")

        # Initialize a policy network with learns from the transitions and selects actions
        self.policy_net = DQN(n_observations, n_actions)
        # Initialize the target network, which provides stable targets for training 
        self.target_net = DQN(n_observations, n_actions)
        # Update the target network with the policy one
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.lr = learning_rate
        # Initialize the optimiser
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        # Initialize the replay memory for the agent
        self.replay = ReplayMemory(10000)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.batch_size = batch_size


    def get_action(self, state):
        """Given a state, or a batch of states, return the set """
        # Sample a value
        sample = random.random()

        # If the value is higher than our threshold, we use the policy network to select an action
        if sample > self.epsilon_start:
            # Ignore the gradient, we only want a forward computation
            with torch.no_grad():
                if type(state) is np.ndarray:
                    state = torch.from_numpy(state)
                # the policy net's probabilities for each action
                probs_per_action = self.policy_net(state)
                # get the one with the highest probability (over columns)
                likely_action = probs_per_action.max(1)
                # get the indeces of that max value and convert it to a 2d array to match the batch
                # processing of the tensor
                action_idx = likely_action.indices.view(1, 1)

                return action_idx
        
        else:
            # We explore and sample from the action space
            return self.env.action_space.sample()


    def decay_epsilon(self):
        """Reduce the epsilon with respect to the given decay, within the bounds"""
        max(self.epsilon_end, self.epsilon_start - self.epsilon_decay)

    