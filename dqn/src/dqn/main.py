from dqn.replay import Transition, ReplayMemory
from dqn.agent import CartPoleAgent
from torch import nn
from torch.nn import functional as F

import gymnasium as gym
import torch

import random
import matplotlib
import matplotlib.pyplot as plt


def cart_pole_env(seed=None):
    """Sets up the CartPole environment from gymnasium as well as seeds
    all the necessary modules with the desired seed (if any)
    
    Args:
        seed: Desired seed to start the RNG in gymnasium and torch
    """
    env = gym.make("CartPole-v1")
    if seed and isinstance(seed, int):
        torch.manual_seed(seed)
        random.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    # number of episodes to run
    n_episodes = 500
    # keeps track of the reward across episodes
    total_rewards = []

    episode_durations = []

    def plot_durations(show_result=False):
        plt.figure(1)
        # Convert durations to a tensor
        durations_t = torch.tensor(episode_durations, dtype=torch.float)

        if show_result:
            plt.title("Result")
        else:
            # Clear the current figure
            plt.clf()
            plt.title("Training...")

        plt.xlabel('Episode')
        plt.ylabel('Duration')

        plt.plot(durations_t.numpy())

        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            # Create slices of 100 elements with a sliding window of 1, resulting in a 2-dimnesional
            # array, compute their mean and flatten the result in a 1d array
            # First slice: idx[0]...idx[99]
            # Second slice: idx[1]...idx[100]
            # Last slice: idx[len(durations)-100]...idx[len(durations)-1]
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            # Prepend the means with zeros such that the plot will start at zero and not at the
            # first value in the means
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())


    agent = CartPoleAgent(env)
    
    total_steps = 0
    # Each episode
    for episode_idx in range(n_episodes):
        # Keep track if we are done or not and the number of iterations in each episode
        done = False
        steps = 0

        # Initialize the total reward for this episode for tracking
        episode_reward = 0
        # Reset the environment
        obs, _info = env.reset()
        # Transform the state into a 2-dimensional tensor with one row and 4 columns (4 values in
        # the state) 
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # While the episode is running
        while not done:
            # Sample an action
            action = agent.get_action(state)

            # Take that action in the environment
            obs, reward, terminated, truncated, info = env.step(action.item())
            # Update the total reward for this episode
            episode_reward += reward

            # Convert the reward into tensor
            reward = torch.tensor([reward])

            # We are done with the episode if we terminated or reached maximum steps
            done = terminated or truncated

            steps += 1
            total_steps += 1

            if terminated:
                # If we terminated the episode, we signal it with a next_state of None
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Store the transition in memory
            agent.replay.push(Transition(state, action, next_state, reward))

            # Move the the next state
            state = next_state

            # reduce the exploration rate. It is better to have it here as oppose to between
            # episodes because different episodes have different lengths. Also it reduces the
            # likelyhood of a sudden jump in decay
            agent.decay_epsilon(total_steps)

            # Perform one step of the optimisation and update the target net
            agent.update_agent()

        episode_durations.append(steps)
        plot_durations()

        total_rewards.append(episode_reward)
    
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

    print(total_rewards)

        
