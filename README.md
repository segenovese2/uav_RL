# UAV Reinforcement Learning
### Trajectory Optimisation for Autonomous Flying Base Station via Reinforcement Learning

---

## Overview

This project implements and extends the reinforcement learning framework presented in:

> H. Bayerlein, P. De Kerret and D. Gesbert, "Trajectory Optimization for Autonomous Flying Base Station via Reinforcement Learning"

A reinforcement learning framework for optimizing autonomous UAV trajectories using multiple algorithms. Implements Q-Learning, DQN, PPO, SAC, and A2C agents trained on a custom 15x15 grid environment with communication constraints.
This project explores whether reinforcement learning can teach a UAV to autonomously
navigate an environment, find the best position to serve multiple ground users, and 
return home within a fixed time budget without being explitly told where to go.

The setup is a 15x15 grid world with two stationary users, an obstacle block that
creates signal shadows, and a UAV that starts at the bottom-left corner with 50 steps
to complete its task. The UAV learns purely from the reward signal - better
communication quality means higher reward, flying into obstacles or failing to return
means penalties.

Two environments are compared. The original replicates the baseline from Bayerlein
et al. (2018), where the reward is the minimum rate across both users, naturally drawing
the UAV toward the equidistant midpoint. The improved environment adds explicit navigation
structure, guiding the agent through three phases:reaching the midpoint, dwelling there 
to maximise communication quality, then returning home before time runs out.

Five algorithms are tested across both environments - Q-Learning, DQN, PPO, SAC, and
A2C - to compare how different learning approaches handle the same task, and whether
the additional structure in the improved environment leads to faster, more consistent
learning.

The project provides two environments:

**Original Environment (`uav_env.py`)**
A recreation of the paper's environment. The reward signal is the minimum
rate across both users (min(r1, r2)), which naturally guides agents toward the equidistant
midpoint between users rather than clustering near a single user. A safety return system
overrides the agent's action when the remaining flight time equals the Manhattan distance
to the start position, forcing the UAV home and applying a penalty for each forced step.
This matches the paper's described safety mechanism.

**Improved Environment (`uav_env_improved.py`)**
An extended version using three-phase navigation structure:
  - Phase 0: Navigate from start to the midpoint between users
  - Phase 1: Dwell at the midpoint, leaving only when the step budget requires return
  - Phase 2: Return to start

Additional reward shaping guides the agent through each phase. The observation space
is extended from 5 to 9 dimensions, adding target coordinates, phase indicator, and
steps-remaining ratio so agents can condition their behaviour on mission phase and
time budget.

Five RL algorithms are compared across both environments:
Q-Learning, DQN, PPO, SAC, and A2C.

---

## Project Structure

```
uav_reinforcement_learning/
├── agents/
│   ├── q_learning_agent.py      # Tabular Q-Learning agent with discretisation
│   ├── wrappers.py              # ContinuousToDiscreteWrapper for SAC
│   └── __init__.py
├── environments/
│   ├── uav_env.py               # Original environment (Bayerlein et al. replication)
│   └── uav_env_improved.py      # Improved environment with 3-phase navigation
├── Results/
│   ├── original_env/            # Trained models and histories for original env (produced upon training)
│   │   ├── qlearning_results/
│   │   ├── dqn_results/
│   │   ├── ppo_results/
│   │   ├── sac_results/
│   │   └── a2c_results/
│   └── improved_env/            # Trained models and histories for improved env (produced upon training)
│       ├── qlearning_results/
│       ├── dqn_results/
│       ├── ppo_results/
│       ├── sac_results/
│       └── a2c_results/
├── agent_train.py               # Training script with GUI environment/agent selection
├── agent_test.py                # Testing script with trajectory visualisation and plot
├── plot_results.py              # Plot training learning curves with GUI selection
├── requirements.txt             # Python dependencies
└── README.txt                   # This file
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Gymnasium
- Stable-Baselines3
- NumPy
- Matplotlib
- Pygame

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch gymnasium stable-baselines3 numpy matplotlib pygame shapely
```

---

## Usage

### 1. Training an Agent

```bash
python agent_train.py
```

A GUI window will appear. Select the environment, agent type, and number of
training episodes, then click Start Training.

Training saves to `Results/{env}_env/{agent}_results/`:
- `training_history_{agent}.npz` -- episode rewards and sum-rates for plotting
- `trained_{agent}.zip`          -- model weights (SB3 agents: DQN, PPO, SAC, A2C)
- `trained_q_learning.npz`       -- Q-table (Q-Learning only)

Recommended episode counts:
- Q-Learning (original env):  800,000  (tabular convergence is slow by design)
- Q-Learning (improved env):  100,000
- DQN:                         50,000
- PPO / A2C / SAC:             30,000

### 2. Testing a Trained Agent

```bash
python agent_test.py
```

A GUI window will appear. Select the environment, agent, and whether to show
per-step diagnostics. The agent runs one episode with a live Pygame visualisation,
then saves a trajectory plot PNG to the project directory.

The trajectory plot shows the full path taken, direction arrows, obstacle and NLOS
shading, user positions, midpoint marker (improved env), and start/end markers.

### 3. Plotting Learning Curves

```bash
python plot_results.py
```

A GUI window will appear. Select the environment and tick whichever agents to
include. The script loads the corresponding `.npz` history files and generates
a smoothed learning curve plot saved as
`training_learning_curves_{env}.png`.

---

## Environment Details

### Original Environment (`uav_env.py`)

| Parameter         | Value                        |
|-------------------|------------------------------|
| Grid size         | 15 x 15                      |
| Users             | [4, 12] and [12, 8]          |
| Obstacle          | 2x4 block at [9-10, 3-6]     |
| Flight time       | 50 steps                     |
| Start / landing   | [0, 0]                       |
| Reward            | min(r1, r2) per step         |
| Obstacle penalty  | -5.0                         |
| Safety system     | Forces return when steps remaining <= Manhattan distance to start |
| Safety penalty    | -abs(reward) per forced step |
| Observation dims  | 5                            |
| Action space      | Discrete(4): up/down/left/right |

### Improved Environment (`uav_env_improved.py`)

Inherits all channel model parameters from the original environment and adds:

| Addition                  | Detail                                              |
|---------------------------|-----------------------------------------------------|
| Midpoint                  | Computed from user positions: [8.0, 10.0]           |
| Phase 0 shaping           | Progress reward toward midpoint (scale 2.0/step)    |
| Midpoint arrival bonus    | +25.0 one-time                                      |
| Phase 1 dwell             | +1.0/step while within radius; must_leave enforced  |
| Phase 2 shaping           | Progress reward toward start (scale 15.0/step)      |
| Return bonus              | +100.0 on reaching start                            |
| Phase 0 failure penalty   | -200.0 if midpoint never reached                    |
| Phase 1 failure penalty   | -50.0 if dwell phase not completed                  |
| Observation dims          | 9 (adds target x/y, phase flag, steps remaining)    |

---

## Agent Details

| Agent      | Type            | Parallel Envs | Notes                                        |
|------------|-----------------|---------------|----------------------------------------------|
| Q-Learning | Tabular         | 1             | Uses GridStateWrapper on original env to match paper's (x,y,t) state space |
| DQN        | Deep Q-Network  | 1             | Off-policy, replay buffer                    |
| PPO        | Policy Gradient | 8             | On-policy, parallel envs reduce variance     |
| SAC        | Actor-Critic    | 8             | Continuous action wrapped to discrete        |
| A2C        | Actor-Critic    | 8             | Synchronous advantage estimation             |

### Q-Learning State Space Note

On the original environment, Q-Learning uses a `GridStateWrapper` that converts
the continuous observation to a direct `(x, y, t)` grid state index, giving
`15 * 15 * 50 = 11,250` states. This matches the paper's tabular implementation
exactly and allows full convergence. Without this wrapper the continuous observation
discretisation produces ~248,832 states which is too large to explore within a
practical number of episodes.

On the improved environment, Q-Learning uses the standard continuous observation
discretisation with `num_bins=6`, giving `6^9 ≈ 10M` states.

---

## Hyperparameters

### Q-Learning
- learning_rate:      0.3   (paper value)
- discount_factor:    0.99  (paper value)
- epsilon_decay:      0.9999 (slow -- needed for 800k episode convergence)
- num_bins (improved): 6

### DQN
- learning_rate:         5e-4
- exploration_fraction:  0.5
- exploration_final_eps: 0.05
- buffer_size:           100,000
- gamma:                 0.99
- batch_size:            64

### PPO
- learning_rate:        3e-4
- n_steps:              50
- batch_size:           256
- ent_coef:             0.01
- normalize_advantage:  True

### SAC
- learning_rate:           1e-3
- buffer_size:             50,000
- batch_size:              128
- ent_coef:                auto
- target_update_interval:  1

### A2C
- learning_rate:       1e-3
- n_steps:             50
- ent_coef:            0.05
- normalize_advantage: True
- gamma:               0.95

---

## Troubleshooting

### Out of memory (Q-Learning)
- On the improved env, reduce num_bins (default 6). Do not increase above 7.

---

## References

Environment and reward model based on:
- Bayerlein et al. (2018): UAV trajectory optimisation via reinforcement learning
- Supervisor reference implementation: https://github.com/cfoh/UAV-Trajectory-Optimization
