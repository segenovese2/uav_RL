import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from agents.q_learning_agent import QLearningAgent
from agents.wrappers import ContinuousToDiscreteWrapper


# ------------------------------------------------------------------ #
#  SELECTION DIALOG                                                    #
# ------------------------------------------------------------------ #

def show_selection_dialog():
    """
    Opens a tkinter window letting the user choose environment, agent,
    and number of training episodes before training begins.
    Returns (env_type, agent_type, num_episodes) or raises SystemExit
    if the user closes the window without confirming.
    """
    root = tk.Tk()
    root.title("UAV Training Setup")
    root.resizable(False, False)

    # Centre the window
    root.update_idletasks()
    w, h = 340, 240
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")

    pad = {"padx": 12, "pady": 6}

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

    # Episodes
    tk.Label(root, text="Episodes:", font=("Segoe UI", 10, "bold")).grid(
        row=2, column=0, sticky="w", **pad)
    episodes_var = tk.StringVar(value="30000")
    episodes_entry = ttk.Entry(root, textvariable=episodes_var, width=20)
    episodes_entry.grid(row=2, column=1, **pad)

    result = {}

    def on_confirm():
        try:
            episodes = int(episodes_var.get())
            if episodes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Episodes must be a positive integer.")
            return
        result["env_type"]    = env_var.get()
        result["agent_type"]  = agent_var.get()
        result["num_episodes"] = episodes
        root.destroy()

    def on_close():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    ttk.Button(root, text="Start Training", command=on_confirm).grid(
        row=4, column=0, columnspan=2, pady=10)

    root.mainloop()

    if not result:
        print("Training cancelled.")
        sys.exit(0)

    return result["env_type"], result["agent_type"], result["num_episodes"]


# ------------------------------------------------------------------ #
#  HELPERS                                                             #
# ------------------------------------------------------------------ #

def get_results_dir(agent_type, env_type):
    folder = os.path.join("Results", f"{env_type}_env", f"{agent_type.lower()}_results")
    os.makedirs(folder, exist_ok=True)
    return folder


# ------------------------------------------------------------------ #
#  CALLBACK                                                            #
# ------------------------------------------------------------------ #

class TrackingCallback(BaseCallback):
    """
    Tracks per-episode average sum-rate and total reward.
    Works correctly with vectorised environments (n_envs > 1).
    """
    def __init__(self, n_envs=1, print_every=100):
        super().__init__()
        self.n_envs = n_envs
        self.print_every = print_every

        self.episode_rewards  = []
        self.episode_sum_rates = []

        self._current_reward   = np.zeros(n_envs)
        self._current_sum_rate = np.zeros(n_envs)
        self._current_steps    = np.zeros(n_envs, dtype=int)

    def _on_step(self):
        rewards = self.locals["rewards"]
        infos   = self.locals["infos"]
        dones   = self.locals["dones"]

        for idx in range(self.n_envs):
            self._current_reward[idx]   += rewards[idx]
            self._current_steps[idx]    += 1
            if "sum_rate" in infos[idx]:
                self._current_sum_rate[idx] += infos[idx]["sum_rate"]

            if dones[idx]:
                self.episode_rewards.append(float(self._current_reward[idx]))
                steps = max(1, self._current_steps[idx])
                self.episode_sum_rates.append(
                    float(self._current_sum_rate[idx]) / steps
                )

                self._current_reward[idx]   = 0
                self._current_sum_rate[idx] = 0
                self._current_steps[idx]    = 0

                n = len(self.episode_rewards)
                if n % self.print_every == 0:
                    avg_r  = np.mean(self.episode_rewards[-self.print_every:])
                    avg_sr = np.mean(self.episode_sum_rates[-self.print_every:])
                    print(f"Episode {n} | Avg Reward: {avg_r:.2f} | "
                          f"Avg Sum-Rate/step: {avg_sr:.4f}")
        return True


# ------------------------------------------------------------------ #
#  TRAIN                                                               #
# ------------------------------------------------------------------ #

def train(agent_type, env_type, num_episodes):

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
    print(f"Episodes    : {num_episodes}")
    print(f"Results dir : {results_dir}\n")

    # ----------------------------------------------------------------
    # Q-LEARNING
    # ----------------------------------------------------------------
    if agent_type == "QLEARNING":
        env = UAVEnv(grid_size=15, render_mode=None)
        env.reset()

        print(f"Training Q-Learning agent for {num_episodes} episodes...")

        # num_bins=6 keeps Q-table manageable for 9-dim improved env
        # (6^9 = ~10M states vs 12^9 = ~5B which causes OOM)
        num_bins = 6 if env_type == "improved" else 12

        agent = QLearningAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            num_bins=num_bins,
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.995
        )

        episode_rewards   = []
        episode_sum_rates = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward   = 0
            total_sum_rate = 0
            steps          = 0
            terminated     = False
            truncated      = False

            while not (terminated or truncated):
                action = agent.choose_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated or truncated)

                if "sum_rate" in info:
                    total_sum_rate += info["sum_rate"]

                state         = next_state
                total_reward += reward
                steps        += 1

            agent.decay()
            episode_rewards.append(total_reward)
            episode_sum_rates.append(total_sum_rate / max(1, steps))

            if (episode + 1) % 100 == 0:
                avg_reward   = np.mean(episode_rewards[-100:])
                avg_sum_rate = np.mean(episode_sum_rates[-100:])
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward (last 100): {avg_reward:.2f} | "
                      f"Avg Sum-Rate/step: {avg_sum_rate:.4f} | "
                      f"epsilon={agent.epsilon:.3f}")

        env.close()

        model_path = os.path.join(results_dir, "trained_q_learning")
        agent.save(model_path)
        print(f"Saved Q-Learning agent to {model_path}.npz")

        history_path = os.path.join(results_dir, "training_history_qlearning.npz")
        np.savez(history_path, sum_rates=episode_sum_rates, rewards=episode_rewards)
        print(f"Saved training history to {history_path}")
        return agent

    # ================================================================
    # SB3 AGENTS
    # ================================================================
    else:
        # DQN gets a larger timestep budget to compensate for n_envs=1
        if agent_type == "DQN":
            num_timesteps = num_episodes * 50 * 8
        else:
            num_timesteps = num_episodes * 50

        n_envs = 8 if agent_type in ("A2C", "PPO", "SAC") else 1

        def make_env():
            env = UAVEnv(grid_size=15, render_mode=None)
            if agent_type == "SAC":
                env = ContinuousToDiscreteWrapper(env)
            return env

        env = make_vec_env(make_env, n_envs=n_envs)

        if agent_type == "DQN":
            model = DQN(
                "MlpPolicy", env,
                verbose=0,
                learning_rate=5e-4,
                exploration_fraction=0.7,
                exploration_final_eps=0.1,
                buffer_size=100000,
                learning_starts=5000,
                gamma=0.99,
                batch_size=64,
            )

        elif agent_type == "PPO":
            model = PPO(
                "MlpPolicy", env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=50,
                batch_size=256,
                ent_coef=0.01,
                normalize_advantage=True,
            )

        elif agent_type == "A2C":
            model = A2C(
                "MlpPolicy", env,
                verbose=0,
                learning_rate=1e-3,
                n_steps=50,
                ent_coef=0.05,
                normalize_advantage=True,
                gamma=0.95,
            )

        elif agent_type == "SAC":
            model = SAC(
                "MlpPolicy", env,
                verbose=0,
                learning_rate=1e-3,
                buffer_size=50000,
                batch_size=128,
                ent_coef='auto',
                target_update_interval=1,
            )

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        callback = TrackingCallback(n_envs=n_envs, print_every=100)

        print(f"Training {agent_type} for ~{num_episodes} episodes "
              f"({num_timesteps} timesteps, {n_envs} parallel envs)...")

        model.learn(total_timesteps=num_timesteps, callback=callback)
        env.close()

        model_path = os.path.join(results_dir, f"trained_{agent_type.lower()}")
        model.save(model_path)
        print(f"Saved {agent_type} model to {model_path}")

        history_path = os.path.join(results_dir, f"training_history_{agent_type.lower()}.npz")
        np.savez(
            history_path,
            sum_rates=np.array(callback.episode_sum_rates),
            rewards=np.array(callback.episode_rewards)
        )
        print(f"Saved training history to {history_path}")
        return model


# ------------------------------------------------------------------ #
#  ENTRY POINT                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    env_type, agent_type, num_episodes = show_selection_dialog()
    train(agent_type, env_type, num_episodes)
