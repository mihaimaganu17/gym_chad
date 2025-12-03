from collections import namedtuple, deque
import random

# A mapping of the current state and action (time-step t) and the respective next state and reward
# after taking the action (time-step t+1)
# A `Transition` which contains a `None` next_state means that after taking the `action`, the agent
# terminated the challenge for the environment
Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])


class ReplayMemory(object):
    """A cyclic buffer of bounded capacity storing the transitions."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    
    def push(self, transition: Transition):
        """Push a new element to the replay memory"""
        self.memory.append(transition)


    def sample(self, batch_size: int) -> [Transition]:
        """Sample `batch_size` count of transitions from the replay memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

