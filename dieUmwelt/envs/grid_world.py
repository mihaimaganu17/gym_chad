from enum import Enum

import gymnasium as gym
import numpy as np


# All possible actions that the agent could take
class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridWorldEnv(gym.Env):
    # This attribute specifies the supported render modes and the framerate at which the environment
    # should be rendered. Every environment should support `None` as render-mode, which does not
    # have to be added in the list
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        """Initialize the environment
        
        Args:
            render_mode: How to render the environment. Supported values are: `None`, `human` and
                `rgb_array`
            size: The size of the square grid.
        """ 
        # Make sure we have a supported render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.size = size
        self.window_size = 512  # The size of the PyGame window

        # The observation space is a dictionary of the agent's and the target's location.
        # Each location is encoded as an element of {0,...,`size`}^2,
        # Both the agent's and the target's location are identically bounded for each dimension.
        # (they move the same amount and have the same boundaries vertically and horizontally)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size-1, shape=(2,), dtype=int),
            }
        )

        # Initial location of the agent and the target
        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)

        # Map the abstract actions from `self.action_space` corresponding to the order of `Actions`
        # to the direction we will walk in if the action is taken.
        self._action_to_direction = {
            # +1 on the x-axis
            Actions.RIGHT.value: np.array([1, 0]),
            # +1 on the y-axis
            Actions.UP.value: np.array([0, 1]),
            # -1 on the x-axis
            Actions.LEFT.value: np.array([-1, 0]),
            # -1 on the y-axis
            Actions.RIGHT.value: np.array([0, -1]),
        }

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self):
        """Return the current observation of the environment"""
        return {"agent": self._agent_location, "target": self._target_location}
    

    def _get_info(self):
        """Returns extra information in the `info` field. In this environment, it gets the taxicab
        distance between the agent and the target"""
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    