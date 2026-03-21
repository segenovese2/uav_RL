import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN, PPO, SAC, A2C
from gymnasium import spaces
from agents.q_learning_agent import QLearningAgent
from agents.wrappers import ContinuousToDiscreteWrapper


# ------------------------------------------------------------------ #
#  GRID STATE WRAPPER (mirrors agent_train.py -- used for Q-learning) #
# ------------------------------------------------------------------ #

class GridStateWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = spaces.Discrete(15 * 15 * 50)
        self.action_space      = env.action_space

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._get_state(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self):
        x = int(self.env.current_pos[0])
        y = int(self.env.current_pos[1])
        t = min(int(self.env.steps), 49)
        return x * 15 * 50 + y * 50 + t

    def close(self):
        self.env.close()

    @property
    def current_pos(self):
        return self.env.current_pos

    @property
    def trajectory(self):
        return self.env.trajectory

    @property
    def nlos_both(self):
        return self.env.nlos_both

    @property
    def nlos_single(self):
        return self.env.nlos_single

    @property
    def obstacles(self):
        return self.env.obstacles

    @property
    def users(self):
        return self.env.users

    @property
    def start_pos(self):
        return self.env.start_pos

    @property
    def grid_size(self):
        return self.env.grid_size

    def render(self):
        return self.env.render()


# ------------------------------------------------------------------ #
#  SELECTION DIALOG                                                    #
# ------------------------------------------------------------------ #

def show_selection_dialog():
    """
    Opens a tkinter window letting the user choose environment and agent
    before the test run begins.
    Returns (env_type, agent_type) or raises SystemExit if cancelled.
    """
    root = tk.Tk()
    root.title("UAV Test Setup")
    root.resizable(False, False)

    root.update_idletasks()
    w, h = 320, 190
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")

    pad = {"padx": 12, "pady": 8}

    # Environment
    tk.Label(root, text="Environment:", font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, sticky="w", **pad)
    env_var = tk.StringVar(value="improved")
    env_combo = ttk.Combobox(root, textvariable=env_var, state="readonly", width=18,
                              values=["original", "improved"])
    env_combo.grid(row=0, column=1, **pad)

    # Agent
    tk.Label(root, text="Agent:", font=("Segoe UI", 10, "bold")).grid(
        row=1, column=0, sticky="w", **pad)
    agent_var = tk.StringVar(value="SAC")
    agent_combo = ttk.Combobox(root, textvariable=agent_var, state="readonly", width=18,
                                values=["QLEARNING", "DQN", "PPO", "SAC", "A2C"])
    agent_combo.grid(row=1, column=1, **pad)

    # Diagnostics toggle
    tk.Label(root, text="Show Diagnostics:", font=("Segoe UI", 10, "bold")).grid(
        row=2, column=0, sticky="w", **pad)
    diag_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(root, variable=diag_var).grid(row=2, column=1, sticky="w", **pad)

    result = {}

    def on_confirm():
        result["env_type"]         = env_var.get()
        result["agent_type"]       = agent_var.get()
        result["show_diagnostics"] = diag_var.get()
        root.destroy()

    def on_close():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    ttk.Button(root, text="Run Test", command=on_confirm).grid(
        row=3, column=0, columnspan=2, pady=12)

    root.mainloop()

    if not result:
        print("Test cancelled.")
        sys.exit(0)

    return result["env_type"], result["agent_type"], result["show_diagnostics"]


# ------------------------------------------------------------------ #
#  TRAJECTORY PLOT                                                     #
# ------------------------------------------------------------------ #

def save_trajectory_plot(env, agent_type, env_type):
    gs  = env.grid_size
    fig, ax = plt.subplots(figsize=(7, 7))

    # NLOS shading
    for cell in env.nlos_both:
        ax.add_patch(patches.Rectangle((cell[0], cell[1]), 1, 1,
                                        color='#787878', zorder=1))
    for cell in env.nlos_single:
        ax.add_patch(patches.Rectangle((cell[0], cell[1]), 1, 1,
                                        color='#b4b4b4', zorder=1))

    # Obstacles
    for obs in env.obstacles:
        ax.add_patch(patches.Rectangle((obs[0], obs[1]), 1, 1,
                                        color='#505050', zorder=2))

    # Midpoint (improved env only)
    if env_type == "improved":
        ax.plot(env.midpoint[0], env.midpoint[1], 'g^',
                markersize=10, zorder=5, label='Midpoint')

    # Users
    for i, user in enumerate(env.users):
        ax.plot(user[0] + 0.5, user[1] + 0.5, 'bo', markersize=12, zorder=5)
        ax.text(user[0] + 0.5, user[1] + 0.5, 'UE', color='white',
                ha='center', va='center', fontsize=7, fontweight='bold', zorder=6)

    # Trajectory
    traj = np.array(env.trajectory)
    ax.plot(traj[:, 0] + 0.5, traj[:, 1] + 0.5,
            'k-', linewidth=2, zorder=4, label='Path')

    # Direction arrows -- placed every few steps so they don't overlap
    arrow_interval = max(1, len(traj) // 10)
    for i in range(0, len(traj) - 1, arrow_interval):
        x  = traj[i, 0] + 0.5
        y  = traj[i, 1] + 0.5
        dx = (traj[i+1, 0] - traj[i, 0]) * 0.4
        dy = (traj[i+1, 1] - traj[i, 1]) * 0.4
        if dx != 0 or dy != 0:
            ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        zorder=7)

    # Start and end markers
    ax.plot(traj[0, 0] + 0.5, traj[0, 1] + 0.5,
            'rs', markersize=10, zorder=6, label='Start')
    ax.plot(traj[-1, 0] + 0.5, traj[-1, 1] + 0.5,
            'r*', markersize=14, zorder=6, label='End')

    ax.set_xlim(0, gs)
    ax.set_ylim(0, gs)
    ax.set_xticks(range(gs + 1))
    ax.set_yticks(range(gs + 1))
    ax.grid(True, linewidth=0.5, color='#cccccc', zorder=0)
    ax.set_title(f"Trajectory  --  {agent_type} on {env_type} environment", fontsize=12)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    filename = f"trajectory_{agent_type}_{env_type}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved trajectory plot to {filename}")
    plt.close()


# ------------------------------------------------------------------ #
#  HELPERS                                                             #
# ------------------------------------------------------------------ #

def get_results_dir(agent_type, env_type):
    return os.path.join("Results", f"{env_type}_env", f"{agent_type.lower()}_results")


# ------------------------------------------------------------------ #
#  TEST                                                                #
# ------------------------------------------------------------------ #

def run_test(env_type, agent_type, show_diagnostics):

    # Import correct environment based on selection
    if env_type == "original":
        from environments.uav_env import UAVEnv
    elif env_type == "improved":
        from environments.uav_env_improved import UAVEnv
    else:
        raise ValueError("ENV_TYPE must be 'original' or 'improved'")

    results_dir = get_results_dir(agent_type, env_type)
    print(f"\nEnvironment : {env_type}")
    print(f"Agent       : {agent_type}")
    print(f"Loading from: {results_dir}\n")

    env = UAVEnv(grid_size=15, render_mode="human")
    state, _ = env.reset()

    # ===== CUSTOM Q-LEARNING =====
    if agent_type == "QLEARNING":
        # Wrap env for original -- matches GridStateWrapper used in training
        if env_type == "original":
            env = GridStateWrapper(env)
        state, _ = env.reset()

        agent = QLearningAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            num_bins=6 if env_type == "improved" else 12
        )
        model_path = os.path.join(results_dir, "trained_q_learning.npz")
        try:
            agent.load(model_path)
            print(f"Loaded Q-Learning agent from {model_path}")
        except FileNotFoundError:
            print(f"Error: could not find model at {model_path}")
            env.close()
            return

        trajectory   = [env.current_pos.copy()]
        total_reward = 0
        steps        = 0

        try:
            while True:
                action = agent.choose_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                trajectory.append(env.current_pos.copy())
                total_reward += reward
                steps        += 1

                if show_diagnostics:
                    if env_type == "improved":
                        phase_names = {0: "Midpoint", 1: "Dwell", 2: "Return"}
                        phase  = phase_names.get(env.phase, str(env.phase))
                        target = env.midpoint if env.phase in (0, 1) else env.start_pos
                        dist   = np.linalg.norm(env.current_pos - target)
                        print(f"Step {steps}: Phase={phase} | Pos={env.current_pos} | "
                              f"Dist={dist:.2f} | Reward={reward:+.3f}")
                    else:
                        print(f"Step {steps}: Pos={env.current_pos} | Reward={reward:+.3f}")

                env.render()
                time.sleep(0.25)

                if terminated or truncated:
                    print(f"\nEpisode finished! Steps: {steps} | Reward: {total_reward:.2f}")
                    print(f"Final position   : {env.current_pos}")
                    print(f"Distance to start: {np.linalg.norm(env.current_pos - env.start_pos):.2f}")
                    save_trajectory_plot(env, agent_type, env_type)
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted.")
        finally:
            env.close()

    # ===== SB3 AGENTS =====
    else:
        model_path = os.path.join(results_dir, f"trained_{agent_type.lower()}")
        try:
            if agent_type == "DQN":
                model = DQN.load(model_path)
            elif agent_type == "PPO":
                model = PPO.load(model_path)
            elif agent_type == "SAC":
                model = SAC.load(model_path)
            elif agent_type == "A2C":
                model = A2C.load(model_path)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            print(f"Loaded {agent_type} agent from {model_path}")
        except FileNotFoundError:
            print(f"Error: could not find model at {model_path}")
            env.close()
            return

        trajectory   = [env.current_pos.copy()]
        total_reward = 0
        steps        = 0

        try:
            while True:
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                trajectory.append(env.current_pos.copy())
                total_reward += reward
                steps        += 1

                if show_diagnostics:
                    if env_type == "improved":
                        phase_names = {0: "Midpoint", 1: "Dwell", 2: "Return"}
                        phase  = phase_names.get(env.phase, str(env.phase))
                        target = env.midpoint if env.phase in (0, 1) else env.start_pos
                        dist   = np.linalg.norm(env.current_pos - target)
                        print(f"Step {steps}: Phase={phase} | Pos={env.current_pos} | "
                              f"Dist={dist:.2f} | Reward={reward:+.3f}")
                    else:
                        print(f"Step {steps}: Pos={env.current_pos} | Reward={reward:+.3f}")

                env.render()
                time.sleep(0.25)

                if terminated or truncated:
                    print(f"\nEpisode finished! Steps: {steps} | Reward: {total_reward:.2f}")
                    print(f"Final position   : {env.current_pos}")
                    print(f"Distance to start: {np.linalg.norm(env.current_pos - env.start_pos):.2f}")
                    save_trajectory_plot(env, agent_type, env_type)
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted.")
        finally:
            env.close()


# ------------------------------------------------------------------ #
#  ENTRY POINT                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",         type=str, default=None)
    parser.add_argument("--agent",       type=str, default=None)
    parser.add_argument("--diagnostics", type=str, default="true")
    args = parser.parse_args()

    if args.env and args.agent:
        show_diag = args.diagnostics.lower() != "false"
        run_test(args.env, args.agent, show_diag)
    else:
        env_type, agent_type, show_diagnostics = show_selection_dialog()
        run_test(env_type, agent_type, show_diagnostics)
