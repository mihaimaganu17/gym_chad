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

    CartPoleAgent(env)
    
