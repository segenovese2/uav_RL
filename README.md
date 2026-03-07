# UAV Reinforcement Learning

A reinforcement learning framework for optimizing autonomous UAV trajectories using multiple algorithms. Implements Q-Learning, DQN, PPO, SAC, and A2C agents trained on a custom 15x15 grid environment with communication constraints.

## Project Structure
uav_reinforcement_learning/
├── agents/
│   ├── q_learning_agent.py      # Q-Learning agent
│   └── init.py
├── environments/
│   └── uav_env.py               # Custom UAV environment (Gym-compatible)
├── test_train.py                # Training script
├── sb3_test.py                  # Testing script (SB3 + Q-Learning)
├── plot_results.py              # Plot training learning curves
├── wrappers.py                  # Action space wrappers (SAC)
└── README.txt                   # This file

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Gymnasium (formerly OpenAI Gym)
- Stable-Baselines3
- NumPy
- Matplotlib
- Pygame (for visualization)

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch gymnasium stable-baselines3 numpy matplotlib pygame
```

## Usage

### 1. Training an Agent

Edit `agent_train.py` to select which agent to train:
```python
AGENT_TYPE = "PPO"  # Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
```

Then run:
```bash
python agent_train.py
```

Training outputs:
- `training_history_{agent}.npz` - Episode rewards and sum-rates
- `trained_{agent}.zip`          - Model weights (SB3 agents)
- `trained_q_learning.npy`       - Q-table (Q-Learning only)

By default, training runs for 30,000 episodes. You can change this by editing the
bottom of `agent_train.py`:
```python
train(AGENT_TYPE, num_episodes=30000)
```

### 2. Testing a Trained Agent

Edit `agent_test.py` to select which agent to test:
```python
AGENT_TYPE = "PPO"  # Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
```

Then run:
```bash
python agent_test.py
```

The agent will run one episode with visualization showing the UAV trajectory.

### 3. Plotting Learning Curves

Edit `plot_results.py` to select which agents to plot:
```python
AGENT_TYPES = ["PPO", "A2C"]  # Any combination of trained agents
```

Then run:
```bash
python plot_results.py
```

This loads the corresponding `training_history_{agent}.npz` files and generates
`training_learning_curves.png` with smoothed learning curves and a summary table.

## Environment Details

### UAV Environment (`environments/uav_env.py`)

Grid: 15x15 continuous space
Agents: 2 stationary users at fixed positions
UAV Task:
- Visit midpoint between users (balanced signal)
- Return to start within 50 steps
- Maximize communication quality

Reward Structure:
- Base:     Sum-rate (log of signal-to-noise ratio)
- Penalties:
    - NLOS (no line-of-sight): -0.5 to -1.0
    - Step penalty:            -0.01
    - On user position:        -5.0
- Bonuses:
    - Proximity to midpoint:   +2.0
    - Visiting midpoint:       +5.0
    - Midpoint + return:       +20.0

Observation Space:
- UAV position (x, y)
- User 1 position and LOS status
- User 2 position and LOS status
- Distance to midpoint
- Step counter

Action Space: 9 discrete actions (8 cardinal directions + stay)

## Agent Details

| Agent      | Type           | Parallel Envs | Best For             |
|------------|----------------|---------------|----------------------|
| Q-Learning | Tabular        | 1             | Small state spaces   |
| DQN        | Deep RL        | 1             | Continuous states    |
| PPO        | Policy Gradient| 8             | Complex tasks        |
| SAC        | Off-Policy     | 1             | Sample efficiency    |
| A2C        | Actor-Critic   | 8             | Stable learning      |

Note: A2C and PPO use 8 parallel environments by default to reduce variance.
SAC uses a ContinuousToDiscreteWrapper (see `wrappers.py`) to bridge its
continuous action output to the environment's discrete action space.

## Hyperparameters

### Q-Learning
- learning_rate:    0.1
- discount_factor:  0.95
- epsilon_decay:    0.995

### DQN
- learning_rate:         1e-3
- exploration_fraction:  0.3
- exploration_final_eps: 0.05

### PPO
- learning_rate:        3e-4
- n_steps:              50
- batch_size:           256
- ent_coef:             0.01
- normalize_advantage:  True

### SAC
- learning_rate:           3e-4
- buffer_size:             10000
- batch_size:              256
- ent_coef:                auto
- target_update_interval:  1

### A2C
- learning_rate:       1e-3
- n_steps:             50
- ent_coef:            0.05
- normalize_advantage: True
- gamma:               0.95

## Troubleshooting

### Agent not learning / stuck in place
- Q-Learning: Increase epsilon (exploration rate)
- DQN:        Reduce exploration_fraction decay, increase buffer_size
- PPO/A2C:    Increase ent_coef
- SAC:        Check ent_coef is set to 'auto' or increase manually

### Out of memory
- Reduce buffer_size (DQN/SAC)
- Reduce batch_size
- Reduce n_envs in test_train.py

### Training too slow
- Reduce num_episodes for a quick test run
- DQN or SAC typically converge faster than PPO

## Key Files Explained

### `agent_train.py`
Main training script. Handles agent initialisation, the episode loop, model
saving, and logging via TrackingCallback (for SB3 agents) or a manual loop
(for Q-Learning).

### `agent_test.py`
Loads a trained model and runs a single visualised episode. Works with all
agent types including Q-Learning.

### `plot_results.py`
Loads `training_history_{agent}.npz` files and plots smoothed learning curves
(average sum-rate per step). Prints a summary table showing first/last 100
episode averages and total improvement.

### `wrappers.py`
Contains ContinuousToDiscreteWrapper, which wraps the discrete UAV environment
with a continuous Box action space so SAC can be used. Actions are converted
back to discrete by taking the argmax of SAC's output.

### `environments/uav_env.py`
Custom Gym-compatible environment with pathloss modelling, Rayleigh fading,
LOS/NLOS propagation, obstacle detection, and reward computation.

## References

uav_env Reward model based on:
- Bayerlein et al. (2018): UAV trajectory optimisation via reinforcement learning

Algorithms:
- Q-Learning
- DQN
- PPO
- SAC
- A2C
