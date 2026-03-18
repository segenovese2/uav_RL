import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN, PPO, SAC, A2C
from agents.q_learning_agent import QLearningAgent
from agents.wrappers import ContinuousToDiscreteWrapper


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
        agent = QLearningAgent(
            observation_space=env.observation_space,
            action_space=env.action_space
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
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted.")
        finally:
            env.close()


# ------------------------------------------------------------------ #
#  ENTRY POINT                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    env_type, agent_type, show_diagnostics = show_selection_dialog()
    run_test(env_type, agent_type, show_diagnostics)
