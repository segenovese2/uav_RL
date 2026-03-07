# UAV Reinforcement Learning

A reinforcement learning framework for optimising autonomous UAV trajectories over a 15x15 grid environment with wireless communication constraints. Supports Q-Learning, DQN, PPO, SAC, and A2C agents via a custom Gymnasium-compatible environment.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Plotting Results](#plotting-results)
- [Environment](#environment)
- [Agents](#agents)
- [Hyperparameters](#hyperparameters)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

The UAV is tasked with navigating a grid world to maximise the sum-rate (communication quality) to two stationary ground users. The environment models path loss, Rayleigh fading, and line-of-sight/NLOS conditions. Two environment variants are provided:

- `uav_env.py` — base environment with comms reward only
- `uav_env_improved.py` — adds navigation shaping to guide the UAV to a midpoint and back to start

---

## Project Structure

```
.
├── agents/
│   ├── q_learning_agent.py      # Tabular Q-Learning agent
│   └── __init__.py
├── environments/
│   ├── uav_env.py               # Base environment
│   └── uav_env_improved.py      # Improved Gymnasium environment
├── agent_train.py               # Training script (SB3 + Q-Learning)
├── agent_test.py                # Testing / visualisation script
├── plot_results.py              # Learning curve plotter
├── wrappers.py                  # Continuous-to-discrete action wrapper (for SAC)
├── requirements.txt             # Requirements before running files
└── README.md
```

---

## Installation

**Requirements:** Python 3.8+

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch gymnasium stable-baselines3 numpy matplotlib pygame
```

---

## Usage

### Training

Open `agent_train.py` and set the agent you want to train:

```python
AGENT_TYPE = "PPO"  # Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
```

Then run:

```bash
python agent_train.py
```

Training outputs saved to the working directory:

| File | Contents |
|------|----------|
| `trained_<agent>.zip` | Model weights (SB3 agents) |
| `trained_q_learning.npy` | Q-table (Q-Learning only) |
| `training_history_<agent>.npz` | Episode rewards and sum-rates |

By default, training runs for 30,000 episodes. You can change this:

```python
train(AGENT_TYPE, num_episodes=5000)
```

### Testing

Open `agent_test.py` and set the agent to test:

```python
AGENT_TYPE = "PPO"  # Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
```

Then run:

```bash
python agent_test.py
```

This loads the saved model, runs one episode with Pygame visualisation, and prints the total reward and step count.

### Plotting Results

After training, open `plot_results.py` and select which agent to plot:

```python
AGENT_TYPES = ["PPO", "A2C"]
```

Then run:

```bash
python plot_results.py
```

This produces `training_learning_curves.png` showing raw and smoothed reward curves, plus a summary table of first/last 100 episode averages.

---

## Environment

### Base Environment (`uav_env.py`)

| Property | Value |
|----------|-------|
| Grid size | 15x15 |
| Action space | Discrete(4) — up, down, left, right |
| Observation space | Box(5) — UAV position, user distances, step ratio |
| Max steps per episode | 50 |
| UAV flight height | 5.0 (grid units) |

**Reward:**
- Each step: sum-rate computed from path loss + Rayleigh fading + LOS/NLOS shadowing
- Obstacle collision: -5.0
- No terminal reward

**Channel model:**

$$R_k = \log_2\left(1 + \frac{P}{N} \cdot d_k^{-\alpha} \cdot h_k \cdot \beta_k\right)$$

where $d_k$ is 3D distance, $h_k$ is Rayleigh fading, and $\beta_k = 1$ for LOS, $0.01$ for NLOS.

### Improved Environment (`uav_env_improved.py`)

Extends the base environment with:

- **Navigation shaping** — per-step progress reward for closing distance to the current target
- **Two-phase mission** — Phase 0: reach midpoint; Phase 1: return to start
- **Milestone bonuses** — one-time reward on reaching the midpoint (+8) and completing the return (+12)
- **Extended observation** — Box(8) adds target coordinates and phase flag so the policy can condition on which leg it is flying

Shaping rewards are intentionally kept in the same scale as the comms reward so they guide without dominating.

---

## Agents

| Agent | Algorithm | Parallel Envs | Notes |
|-------|-----------|--------------|-------|
| `QLEARNING` | Tabular Q-Learning | 1 | Fast, works well on small state spaces |
| `DQN` | Deep Q-Network | 1 | Off-policy, experience replay |
| `PPO` | Proximal Policy Optimisation | 8 | Stable, good for longer horizons |
| `A2C` | Advantage Actor-Critic | 8 | Fast updates, benefits from parallel envs |
| `SAC` | Soft Actor-Critic | 1 | Uses `ContinuousToDiscreteWrapper` to bridge action spaces |

SB3 agents (DQN, PPO, A2C, SAC) are implemented using [Stable Baselines3](https://stable-baselines3.readthedocs.io/).

---

## Hyperparameters

### Q-Learning
| Parameter | Default | Effect |
|-----------|---------|--------|
| `learning_rate` | 0.1 | Higher = faster but less stable |
| `discount_factor` | 0.95 | Future reward weighting |
| `exploration_rate` | 1.0 | Starting epsilon |
| `exploration_decay` | 0.995 | Epsilon decay per episode |

### DQN
| Parameter | Default |
|-----------|---------|
| `learning_rate` | 1e-3 |
| `exploration_fraction` | 0.3 |
| `exploration_final_eps` | 0.05 |

### PPO
| Parameter | Default |
|-----------|---------|
| `learning_rate` | 3e-4 |
| `n_steps` | 50 |
| `batch_size` | 256 |
| `ent_coef` | 0.01 |

### A2C
| Parameter | Default |
|-----------|---------|
| `learning_rate` | 1e-3 |
| `n_steps` | 50 |
| `ent_coef` | 0.05 |
| `gamma` | 0.95 |

### SAC
| Parameter | Default |
|-----------|---------|
| `learning_rate` | 3e-4 |
| `buffer_size` | 10000 |
| `batch_size` | 256 |
| `ent_coef` | `'auto'` |

---

## Troubleshooting

**Agent not moving / stuck in a corner**
- Increase `ent_coef` for PPO/A2C to encourage exploration
- For Q-Learning, slow down `exploration_decay` or raise starting `exploration_rate`
- For DQN, increase `exploration_fraction`

**Reward not improving**
- Check the training history plot — if sum-rate is flat from episode 1, the agent may be camping at a high-reward cell rather than navigating
- Consider switching to `uav_env_improved.py`, which adds explicit navigation guidance

**Out of memory**
- Reduce `buffer_size` for DQN/SAC
- Reduce `batch_size`

**SAC action space error**
- SAC requires continuous actions; make sure `ContinuousToDiscreteWrapper` from `wrappers.py` is applied when constructing the environment

---

## References

- Bayerlein et al. (2018), "Trajectory Optimization for Autonomous Flying Base Station via Reinforcement Learning" — reward model basis
- Stable Baselines3 documentation: https://stable-baselines3.readthedocs.io/
- Gymnasium documentation: https://gymnasium.farama.org/
