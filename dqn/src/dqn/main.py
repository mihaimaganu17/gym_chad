from dqn.replay import Transition, ReplayMemory
from dqn.agent import CartPoleAgent
from torch import nn
from torch.nn import functional as F

import gymnasium as gym
import torch

import random


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
    n_episodes = 100000
    # keeps track of the reward across episodes
    total_rewards = []


    agent = CartPoleAgent(env)

    # Each episode
    for _ in range(n_episodes):
        # Keep track if we are done or not and the number of iterations in each episode
        done = False
        steps = 0
        episode_reward = 0

        # While the episode is running
        while not done:
            # Reset the environment
            obs, _info = env.reset()

            # Sample an action
            action = agent.get_action(obs)

            # Take that action on the environment
            obs, reward, terminated, truncated, info = env.step(action.item())
            # Update the total reward for this episode
            episode_reward += reward

            # We are done with the episode if we terminated or reached maximum steps
            done = terminated or truncated

            steps += 1
            
            # reduce the exploration rate
            agent.decay_epsilon()
        

        total_rewards.append(episode_reward)

    print(total_rewards)

        
