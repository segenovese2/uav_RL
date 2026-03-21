import numpy as np
import random
import json
import gymnasium as gym

class QLearningAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        num_bins=6,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01
    ):
        """
        A fully compatible Q-learning agent that works with:
        - Any discrete Gym environment
        - Any continuous Gym environment (auto-discretized)
        - Your custom environment, as long as it returns a vector state

        :param observation_space: env.observation_space
        :param action_space: env.action_space
        """

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate

        self.action_size = action_space.n

        # ------------------------------
        #  Universal State Handling
        # ------------------------------

        if isinstance(observation_space, gym.spaces.Discrete):
            # Example: FrozenLake, Taxi-v3
            self.state_type = "discrete"
            self.state_size = observation_space.n
            self.num_bins = None

        elif isinstance(observation_space, gym.spaces.Box):
            # Example: CartPole, MountainCar, LunarLander
            self.state_type = "continuous"
            self.low = observation_space.low
            self.high = observation_space.high
            self.num_bins = num_bins

            # Each dimension gets num_bins discretization buckets
            self.state_size = (num_bins ** len(observation_space.low))

        else:
            raise ValueError("Unsupported observation space type.")

        # Create Q-table
        self.q_table = np.ones((self.state_size, self.action_size)) * 5.0

    # ------------------------------------------------------------------
    # State Discretization
    # ------------------------------------------------------------------
    def _discretize_state(self, state):
        if self.state_type == "discrete":
            if hasattr(state, 'item'):
                return int(state.item())
            return int(state)

        # Continuous → bucketize each dimension
        ratios = (state - self.low) / (self.high - self.low + 1e-8)
        bins = np.floor(ratios * self.num_bins).astype(int)
        bins = np.clip(bins, 0, self.num_bins - 1)

        # Convert multi-dimensional bin vector → single index
        index = 0
        for i, b in enumerate(bins):
            index += b * (self.num_bins ** i)
        return index

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------
    def choose_action(self, state, training=True):
        state_idx = self._discretize_state(state)

        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        return np.argmax(self.q_table[state_idx])

    # ------------------------------------------------------------------
    # Q-Table Update
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        s = self._discretize_state(state)
        ns = self._discretize_state(next_state)

        target = reward if done else reward + self.gamma * np.max(self.q_table[ns])
        self.q_table[s, action] += self.lr * (target - self.q_table[s, action])

    # ------------------------------------------------------------------
    # Exploration Decay
    # ------------------------------------------------------------------
    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Saving & Loading
    # ------------------------------------------------------------------
    def save(self, filepath):
        """Save Q-table using compressed format for faster I/O."""
        np.savez_compressed(filepath, q_table=self.q_table)

    def load(self, filepath):
        """Load Q-table from compressed format."""
        data = np.load(filepath)
        self.q_table = data['q_table']
