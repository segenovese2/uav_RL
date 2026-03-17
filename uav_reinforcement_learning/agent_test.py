import os
import time
import numpy as np
from stable_baselines3 import DQN, PPO, SAC, A2C
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
AGENT_TYPE = "QLEARNING"  # Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
# ============================

SHOW_DIAGNOSTICS = True  # Enable per-step phase/distance logging

if ENV_TYPE == "original":
    from environments.uav_env import UAVEnv
elif ENV_TYPE == "improved":
    from environments.uav_env_improved import UAVEnv
else:
    raise ValueError("ENV_TYPE must be 'original' or 'improved'")


def get_results_dir(agent_type, env_type):
    return os.path.join("Results", f"{env_type}_env", f"{agent_type.lower()}_results")


def run_test():
    results_dir = get_results_dir(AGENT_TYPE, ENV_TYPE)
    print(f"Loading model from: {results_dir}")

    env = UAVEnv(grid_size=15, render_mode="human")
    state, _ = env.reset()

    # ===== CUSTOM Q-LEARNING =====
    if AGENT_TYPE == "QLEARNING":
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

        trajectory = [env.current_pos.copy()]
        total_reward = 0
        steps = 0

        try:
            while True:
                action = agent.choose_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                trajectory.append(env.current_pos.copy())
                total_reward += reward
                steps += 1

                if SHOW_DIAGNOSTICS:
                    phase_names = {0: "Midpoint", 1: "Dwell", 2: "Return"}
                    phase = phase_names.get(env.phase, str(env.phase))
                    target = env.midpoint if env.phase in (0, 1) else env.start_pos
                    dist = np.linalg.norm(env.current_pos - target)
                    print(f"Step {steps}: Phase={phase} | Pos={env.current_pos} | "
                          f"Dist={dist:.2f} | Reward={reward:+.3f}")

                env.render()
                time.sleep(0.25)

                if terminated or truncated:
                    print(f"\nEpisode finished! Steps: {steps} | Reward: {total_reward:.2f}")
                    print(f"Final position: {env.current_pos}")
                    print(f"Distance to start: {np.linalg.norm(env.current_pos - env.start_pos):.2f}")
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted.")
        finally:
            env.close()

    # ===== SB3 AGENTS =====
    else:
        model_path = os.path.join(results_dir, f"trained_{AGENT_TYPE.lower()}")
        try:
            if AGENT_TYPE == "DQN":
                model = DQN.load(model_path)
            elif AGENT_TYPE == "PPO":
                model = PPO.load(model_path)
            elif AGENT_TYPE == "SAC":
                model = SAC.load(model_path)
            elif AGENT_TYPE == "A2C":
                model = A2C.load(model_path)
            else:
                raise ValueError(f"Unknown agent type: {AGENT_TYPE}")
            print(f"Loaded {AGENT_TYPE} agent from {model_path}")
        except FileNotFoundError:
            print(f"Error: could not find model at {model_path}")
            env.close()
            return

        trajectory = [env.current_pos.copy()]
        total_reward = 0
        steps = 0

        try:
            while True:
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                trajectory.append(env.current_pos.copy())
                total_reward += reward
                steps += 1

                if SHOW_DIAGNOSTICS:
                    phase_names = {0: "Midpoint", 1: "Dwell", 2: "Return"}
                    phase = phase_names.get(env.phase, str(env.phase))
                    target = env.midpoint if env.phase in (0, 1) else env.start_pos
                    dist = np.linalg.norm(env.current_pos - target)
                    print(f"Step {steps}: Phase={phase} | Pos={env.current_pos} | "
                          f"Dist={dist:.2f} | Reward={reward:+.3f}")

                env.render()
                time.sleep(0.25)

                if terminated or truncated:
                    print(f"\nEpisode finished! Steps: {steps} | Reward: {total_reward:.2f}")
                    print(f"Final position: {env.current_pos}")
                    print(f"Distance to start: {np.linalg.norm(env.current_pos - env.start_pos):.2f}")
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted.")
        finally:
            env.close()


if __name__ == "__main__":
    run_test()
