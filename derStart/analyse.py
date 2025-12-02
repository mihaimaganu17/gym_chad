from matplotlib import pyplot as plt
from qagent import BlackjackAgent

import gymnasium as gym
import numpy as np


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode,
    ) / window

def analyse(env: gym.Env, agent: BlackjackAgent):

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid",
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    # Episodes lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same",
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

    plt.tight_layout()
    plt.show()
