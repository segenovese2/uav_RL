import gymnasium as gym
from gymnasium import ActionWrapper
import numpy as np


class ContinuousToDiscreteWrapper(ActionWrapper):
    """
    Wraps a discrete action space environment to accept continuous actions.
    SAC requires continuous actions, so this wrapper converts SAC's continuous
    output into discrete actions by taking the argmax.
    """
    def __init__(self, env):
        super().__init__(env)
        # Store the original discrete action space
        self.discrete_action_space = env.action_space
        # Replace with continuous action space for SAC
        # Create one continuous output per discrete action
        self.action_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(self.discrete_action_space.n,), 
            dtype=np.float32
        )
    
    def action(self, action):
        """Convert continuous action to discrete by taking argmax."""
        # action is shape (4,) from SAC
        # return the index of the maximum value (0-3)
        return np.argmax(action)
