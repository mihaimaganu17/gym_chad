from enum import Enum

import gymnasium as gym
import numpy as np
import pygame


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
    

    def reset(self, seed=None, options=None):
        """Initiate a new episode of the environment

        Args:
            seed: An integer used to seed the RNG such that we get a reproducible results across
                resets
            options: Particularities that can be passed by the user to this specific environment
        """
        # gym.Env uses numpy random generator and it is recommended we use it as well. We reseed it
        # with the given seed
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random. Returns an ndarray of 2 integers
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will also sample the target's location randomly until it does not coincide with the
        # agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            # Resample
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)


        # The first observation and info after reset
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def step(self, action: Actions):
        """Computes the state of the environment after taking the given action and returns a 5-tuple
        that describes the new state of the environment along with a reward

        Args:
            action: One of the 4 possible `Actions` value

        Returns: a 5-tuple containing:
            observation: A dictionary with the `agent` and the `target` location
            reward: A reward for the agent for taking the action (1 if terminated, 0 otherwise)
            terminated: Whether or not the episode is finished
            truncated: Whether of not we reached the maximum allowed steps for the episode
            info: The Manhattan distance between the agent and the target
        """
        # Map the given `action` to the direction the agent is walkin
        direction = self._action_to_direction[action]
        # Compute the new agent location using `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size-1)

        # If the agent reached the target, signal that the episode is done
        terminated = np.array_equal(self._agent_location, self._target_location)
        # Binary sparse rewards
        reward = 1 if terminated else 0
        # Get the new state's observation and the distance
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    
    def _render_frame(self):
        # Initialize the window and pygame
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # Set a display with desired (width, height)
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        # Initialize the clock
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create a new white canvas
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill([0xff, 0xff, 0xff])

        # The size of a single grid square in pixels
        pix_square_size = (self.window_size / self.size)

        # Create the target as a rectangle with location (left, top) and full grid square dimensions (width, height).
        target_rect = pygame.Rect(
            # (left, top)
            (pix_square_size * self._target_location),
            # (width, height)
            (pix_square_size, pix_square_size),
        )

        # Draw a red target on the canvas
        pygame.draw.rect(
            canvas,
            (0xff, 0, 0),
            target_rect,
        )

        # Create the agent as a blue circle and draw it on the canvas
        agent_circle = pygame.draw.circle(
            canvas,
            (0, 0, 0xff),
            # center of the circle (sliding it to the midle from the top left corner of the square)
            (self._agent_location + 0.5) * pix_square_size,
            # The radius of the circle
            pix_square_size / 3,
        )

        # Add some gridlines
        for x in range(self.size + 1):
            # Draw horizontal lines
            pygame.draw.line(
                canvas,
                0,
                [0, x] * pix_square_size,
                [self.size, x] * pix_square_size,
                width=3,
            )
            # Draw vertical lines
            pygame.draw.line(
                canvas,
                0,
                [x, 0] * pix_square_size,
                [x, self.size] * pix_square_size,
                width=3,
            )

        if self.render_mode == "human":
            # Draw the canvas onto the display
            self.window.blit(canvas, canvas.get_rect())
            # Process the pygame event handlers from the event queue
            pygame.event.pump()
            # Update the visible display surface
            pygame.display.update()

            # Making sure we run at the desired FPS
            self.clock.tick(self.metadata["render_fps"])

        else:
            # rgb_array
            # Pygame has the rgb axes stored in the following order (width, height, 3)
            # Whereas numpy, gym render frame, PIL, OpenCV, matplotlib have the (height, widht, 3)
            # order, that is why we transpose
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    
    def close(self):
        # Close the used resources and clean up
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


