from collections import defaultdict
import numpy as np
import gymnasium as gym

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0) 
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards. Used to discount the future q value
        """
        self.env = env

        # Q-table maps (state, action) to expected reward
        # We populate the dict with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        # How much we care about future rewards
        self.dicount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track leaning progress
        self.training_error = []


    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy

        Args:
            obs: Observation tuple with 3 fields:
                player_sum: Current hand value (4-21)
                dealer_card: Dealer’s face-up card (1-10)
                usable_ace: Whether player has usable ace (True/False)

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # With probability (1-epsilon): exploit (best known action)
        else:
            # Return the index of the best possible action
            return (int(np.argmax(self.q_values[obs])))
        
    
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        Args:
            obs: Observation tuple with 3 fields:
                player_sum: Current hand value (4-21)
                dealer_card: Dealer’s face-up card (1-10)
                usable_ace: Whether player has usable ace (True/False)
            action: Action that was taken
            reward: Reward given as a response to the taken action
            terminated: Was the episode terminated?
            next_obs: Observation as a result of the action taken. Has the same structure as `obs`

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """

        # Bellman start
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # What should the Q-value be?
        target = reward + self.dicount_factor * future_q_value
        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]
        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take regarding the information that overrides the
        # old Q-value
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        # Bellman end

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)


    def decay_epsilon(self):
        """Reduce exploration rate after each episode"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)