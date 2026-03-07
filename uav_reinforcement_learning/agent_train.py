import numpy as np
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from agents.q_learning_agent import QLearningAgent
from wrappers import ContinuousToDiscreteWrapper

# ============================
# SELECT Environment
# ============================
#from environments.uav_env import UAVEnv
from environments.uav_env_improved import UAVEnv
# ============================

# ============================
# SELECT AGENT HERE
# ============================
AGENT_TYPE = "QLEARNING"  # Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
# ============================


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
                # store average sum-rate per step so scale is independent of episode length
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


def train(agent_type=AGENT_TYPE, num_episodes=500, show_training=False):

    # ----------------------------------------------------------------
    # Q-LEARNING
    # ----------------------------------------------------------------
    if agent_type == "QLEARNING":
        env = UAVEnv(grid_size=15, render_mode=None)
        env.reset()

        print(f"Training Q-Learning agent for {num_episodes} episodes...")

        agent = QLearningAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
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
            episode_sum_rates.append(total_sum_rate / max(1, steps))  # avg per step

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

        model_path = "trained_q_learning"
        agent.save(model_path)
        print(f"Saved Q-Learning agent to {model_path}.npz")

        np.savez(
            "training_history_qlearning.npz",
            sum_rates=episode_sum_rates,
            rewards=episode_rewards
        )
        print("Saved training history to training_history_qlearning.npz")
        return agent

    # ================================================================
    # SB3 AGENTS
    # ================================================================
    else:
        num_timesteps = num_episodes * 50

        # A2C benefits greatly from parallel envs to reduce variance.
        # PPO also benefits. DQN must use n_envs=1. SAC works with 1 env.
        n_envs = 8 if agent_type in ("A2C", "PPO", "SAC") else 1

        def make_env():
            env = UAVEnv(grid_size=15, render_mode=None)
            # Wrap with continuous action converter if using SAC
            if agent_type == "SAC":
                env = ContinuousToDiscreteWrapper(env)
            return env

        env = make_vec_env(make_env, n_envs=n_envs)

        if agent_type == "DQN":
            model = DQN(
                "MlpPolicy", env,
                verbose=0,
                learning_rate=1e-3,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
            )

        elif agent_type == "PPO":
            model = PPO(
                "MlpPolicy", env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=50,           # matches episode length
                batch_size=256,
                ent_coef=0.01,        # encourages exploration
                normalize_advantage=True,
            )

        elif agent_type == "A2C":
            model = A2C(
                "MlpPolicy", env,
                verbose=0,
                learning_rate=1e-3,
                n_steps=50,           # collect one full episode per update
                ent_coef=0.05,        # higher entropy to stop policy collapsing to corner
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
                ent_coef='auto',      # automatic entropy tuning
                target_update_interval=1,
            )

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        callback = TrackingCallback(n_envs=n_envs, print_every=100)

        print(f"Training {agent_type} for ~{num_episodes} episodes "
              f"({num_timesteps} timesteps, {n_envs} parallel envs)...")

        model.learn(total_timesteps=num_timesteps, callback=callback)

        env.close()

        model_path = f"trained_{agent_type.lower()}"
        model.save(model_path)
        print(f"Saved {agent_type} model to {model_path}")

        np.savez(
            f"training_history_{agent_type.lower()}.npz",
            sum_rates=np.array(callback.episode_sum_rates),
            rewards=np.array(callback.episode_rewards)
        )
        print(f"Saved training history to training_history_{agent_type.lower()}.npz")

        return model


if __name__ == "__main__":
    train(AGENT_TYPE, num_episodes=100000)