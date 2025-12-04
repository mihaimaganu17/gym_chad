from dqn.dqn import DQN
from dqn.replay import ReplayMemory, Transition
from torch import nn

import gymnasium as gym
import torch
import torch.optim as optim
import random
import numpy as np
import math


class CartPoleAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 2500,
        discount_factor: float = 0.99,
        tau: float = 0.005,
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
            tau: How we are blending the weights of the policy network with the ones 
                from the target network
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

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.tau = tau

        self.losses = []


    def get_action(self, state):
        """Given a state, or a batch of states, return the set """
        # Sample a value
        sample = random.random()

        # If the value is higher than our threshold, we use the policy network to select an action
        if sample > self.epsilon:
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
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)


    def decay_epsilon(self, steps):
        """Reduce the epsilon with respect to the given decay, within the bounds"""
        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.epsilon_decay)
     
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * steps / self.epsilon_decay)

    
    def update_agent(self):
        # If there is not enough replay memory to fill the batch, return
        if len(self.replay) < self.batch_size:
            return

        # Sample from the replay memory
        transitions = self.replay.sample(self.batch_size)
        # Transpose the batch from a batch of `Transition`'s to a `Transition` of batches.
        # See https://stackoverflow.com/a/19343/3343043
        batch = Transition(*zip(*transitions))

        # Non-final next states are transitions that don't have a next_state `None` value
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        

        # Concatenate all batch elements into a single 1-dim tensor
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute the Q - value at this current time-step.
        # This computes the probability of the action taken for each instance in the batch
        full_action_probs = self.policy_net(state_batch)
        # Get the values from the probabilities. With gather on dim=1, we basically use action_batch
        # to index the probabilities exported by a policy network forward pass
        # Example:
        #   action_probs = [[0.095, 0.12], [0.144, 0.543]]
        #   action_batch = [[0], [1]]
        #   action_probs.gather(1, action_batch) will be [[0.095], [0.543]]
        actual_action_probs = full_action_probs.gather(1, action_batch)

        # Compute the future value of all next states (time-step: t+1)
        # Create a new zero-filled tensor to hold the values for the entire batch
        next_state_values = torch.zeros(self.batch_size)

        # Compute a mask of non final next states, which mark if transitions have the next_state
        # `None` or not. These are used to index only the next_state values which are not final
        # in order to populate them with results from the target network.
        non_final_states_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        
        # Make sure we don't compute gradients when using the target net
        with torch.no_grad():
            # We compute the expected values for non final states using the "current" target network.
            # We select the best "reward" using max(1).values. Actually these are probabilities of
            # the actions to be taken
            next_state_values[non_final_states_mask] = \
                self.target_net(non_final_next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute the Huber loss
        huber_loss = nn.SmoothL1Loss()
        loss = huber_loss(actual_action_probs, expected_state_action_values.unsqueeze(1))

        self.losses.append(loss)

        # Optimize the model, reset all gradients
        self.optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Clip the gradient to prevent exploding gradients or very low values
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        # Perform a single optimisation step
        self.optimizer.step()

        # Soft update the target network's weights, blending the target weights with the policy ones
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau \
                + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)






            
