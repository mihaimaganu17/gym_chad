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
        # One such action-observation exchange is called a `timestep`
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    print(env.observation_space)
    env.close()


def _sample_timesteps():
    # Discrete action space (button presses)
    env = gym.make("CartPole-v1")
    print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
    print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

    # Box observation space (continuous values)
    print(f"Observation space: {env.observation_space}")  # Box with 4 values
    # Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
    print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation
    env.close()


if __name__ == "__main__":
    main()
