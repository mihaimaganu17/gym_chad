from qagent import BlackjackAgent
from analyse import analyse
from tqdm import tqdm

import numpy as np
import gymnasium as gym


n_episodes = 100_000        # Number of hands to practice

def _envs():
    for i in gym.envs.registry.keys():
        print(i)


def init_env() -> gym.Env:
    """Initialise the gym environment

    Returns: The gym.Env environment
    """
    # the "sab" argument toggles whether the environment uses the rules defined in Sutton & Barto’s
    # Reinforcement Learning textbook.
    # When sab=True, the environment uses the exact Blackjack rules described in the textbook,
    # player sticks only on 20 or 21 (Instead of the more standard rule of sticking on 17+)
    # When sab=Fals, this uses the more standard Casino-style version where player sticks on 17 or
    # higher values and returns follow common blackjack rules instead of the textbook-specific
    # variant.
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    return env


def init_agent(env: gym.Env) -> BlackjackAgent:
    """Initialise the BlackjackAgent
    
    Args:
        env: The gymnasium environment

    Returns: the BlackjackAgent
    """

    learning_rate = 0.001        # How fast to learn (hight = faster but less stable)
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2) # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    return agent


def play_blackjack():
    """Main Q-Learning loop using the Blackjack environment and Agent
    """
    # Initialise the environment
    env = init_env()
    # Initialise the agent
    agent = init_agent(env)

    for episode in tqdm(range(n_episodes)):
        # Start a new hand
        obs, info = env.reset()
        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)
            
            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            obs = next_obs

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()

    # Analyse the results
    analyse(env, agent)

    # Clean up after finishing
    env.close()


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
    BlackjackAgent(env, 0.1, 1.0, 0.1, 0.1)
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
    #main()
    play_blackjack()
