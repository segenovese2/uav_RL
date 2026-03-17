import os
import numpy as np
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from agents.q_learning_agent import QLearningAgent
from agents.wrappers import ContinuousToDiscreteWrapper

# ============================
# SELECT Environment
# ============================
ENV_TYPE = "improved"  # Options: "original", "improved"
# ============================

# ============================
# SELECT AGENT HERE
# ============================
AGENT_TYPE = "SAC"  # Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
# ============================

if ENV_TYPE == "original":
    from environments.uav_env import UAVEnv
elif ENV_TYPE == "improved":
    from environments.uav_env_improved import UAVEnv
else:
    raise ValueError("ENV_TYPE must be 'original' or 'improved'")

# Results folder: Results/original_env/agentname_results/
#              or Results/improved_env/agentname_results/
def get_results_dir(agent_type, env_type):
    folder = os.path.join(
        "Results",
        f"{env_type}_env",
        f"{agent_type.lower()}_results"
    )
    os.makedirs(folder, exist_ok=True)
    return folder


class TrackingCallback(BaseCallback):
    """
    Tracks per-episode average sum-rate and total reward.
    Works correctly with vectorised environments (n_envs > 1).
    """
    def __init__(self, n_envs=1, print_every=100):
        super().__init__()
        self.n_envs = n_envs
        self.print_every = print_every

        self.episode_rewards = []
        self.episode_sum_rates = []

        # accumulators per env
        self._current_reward = np.zeros(n_envs)
        self._current_sum_rate = np.zeros(n_envs)
        self._current_steps = np.zeros(n_envs, dtype=int)

    def _on_step(self):
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for idx in range(self.n_envs):
            self._current_reward[idx] += rewards[idx]
            self._current_steps[idx] += 1
            if "sum_rate" in infos[idx]:
                self._current_sum_rate[idx] += infos[idx]["sum_rate"]

            if dones[idx]:
                self.episode_rewards.append(float(self._current_reward[idx]))
                steps = max(1, self._current_steps[idx])
                self.episode_sum_rates.append(
                    float(self._current_sum_rate[idx]) / steps
                )

                self._current_reward[idx] = 0
                self._current_sum_rate[idx] = 0
                self._current_steps[idx] = 0

                n = len(self.episode_rewards)
                if n % self.print_every == 0:
                    avg_r = np.mean(self.episode_rewards[-self.print_every:])
                    avg_sr = np.mean(self.episode_sum_rates[-self.print_every:])
                    print(
                        f"Episode {n} | "
                        f"Avg Reward: {avg_r:.2f} | "
                        f"Avg Sum-Rate/step: {avg_sr:.4f}"
                    )
        return True


def train(agent_type=AGENT_TYPE, env_type=ENV_TYPE, num_episodes=500, show_training=False):

    results_dir = get_results_dir(agent_type, env_type)
    print(f"Results will be saved to: {results_dir}")

    # ----------------------------------------------------------------
    # Q-LEARNING
    # ----------------------------------------------------------------
    if agent_type == "QLEARNING":
        env = UAVEnv(grid_size=15, render_mode=None)
        env.reset()

        print(f"Training Q-Learning agent for {num_episodes} episodes...")

        # num_bins=6 keeps Q-table size manageable for 9-dim improved env
        # (6^9 = ~10M states vs 12^9 = ~5B states which causes OOM)
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

        episode_rewards = []
        episode_sum_rates = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            total_sum_rate = 0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.choose_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated or truncated)

                if "sum_rate" in info:
                    total_sum_rate += info["sum_rate"]

                state = next_state
                total_reward += reward
                steps += 1

            agent.decay()
            episode_rewards.append(total_reward)
            episode_sum_rates.append(total_sum_rate / max(1, steps))

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_sum_rate = np.mean(episode_sum_rates[-100:])
                print(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Avg Reward (last 100): {avg_reward:.2f} | "
                    f"Avg Sum-Rate/step: {avg_sum_rate:.4f} | "
                    f"epsilon={agent.epsilon:.3f}"
                )

        env.close()

        model_path = os.path.join(results_dir, "trained_q_learning")
        agent.save(model_path)
        print(f"Saved Q-Learning agent to {model_path}.npz")

        history_path = os.path.join(results_dir, "training_history_qlearning.npz")
        np.savez(
            history_path,
            sum_rates=episode_sum_rates,
            rewards=episode_rewards
        )
        print(f"Saved training history to {history_path}")
        return agent

    # ================================================================
    # SB3 AGENTS
    # ================================================================
    else:
        # DQN gets a larger timestep budget to compensate for n_envs=1
        # SAC/PPO/A2C use 8 parallel envs so their effective experience is already higher
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


if __name__ == "__main__":
    train(AGENT_TYPE, ENV_TYPE, num_episodes=2000)
