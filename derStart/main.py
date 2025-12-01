import numpy as np
import gymnasium as gym


def _envs():
    for i in gym.envs.registry.keys():
        print(i)


def main():
    # Create the training environment
    env = gym.make('CartPole-v1', render_mode="human")

    # Reset environment to start a new episode
    observation, info = env.reset()
    # observation: what the agent can "see" - cart position, velocity, pole angle
    # info: extra debugging information
    print(f"Starting observartion: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = env.action_space.sample()

        # Take the actin and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
