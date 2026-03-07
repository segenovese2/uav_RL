import time
import numpy as np
from stable_baselines3 import DQN, PPO, SAC, A2C
from agents.q_learning_agent import QLearningAgent
from wrappers import ContinuousToDiscreteWrapper

# ============================
ENV_TYPE = "improved env"   # options: "base", "improved"
AGENT_TYPE = "QLEARNING"  # Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
SHOW_DIAGNOSTICS = True  # Enable phase/distance logging
# ============================

if ENV_TYPE == "base env":
    from environments.uav_env import UAVEnv
elif ENV_TYPE == "improved env":
    from environments.uav_env_improved import UAVEnv
else:
    raise ValueError("ENV_TYPE must be 'base env' or 'improved env'")

def run_test():
    env = UAVEnv(grid_size=15, render_mode="human")
    state, _ = env.reset()
    
    # ===== CUSTOM Q-LEARNING =====
    if AGENT_TYPE == "QLEARNING":
        agent = QLearningAgent(
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        model_path = f"Results/{ENV_TYPE}/q-learning_results/trained_q_learning.npz"
        try:
            agent.load(model_path)
            print(f"✓ Loaded Q-Learning agent from {model_path}")
        except FileNotFoundError:
            print(f"✗ Error: Could not find model at {model_path}")
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
                    phase = "Midpoint" if env.phase == 0 else "Return"
                    dist_to_target = np.linalg.norm(
                        env.current_pos - (env.midpoint if env.phase == 0 else env.start_pos)
                    )
                    print(f"Step {steps}: Phase={phase} | Pos={env.current_pos} | Dist={dist_to_target:.2f} | Reward={reward:+.3f}")
                
                env.render()
                time.sleep(0.25)
                
                if terminated or truncated:
                    print(f"\n✓ Episode finished!\n  Steps: {steps} | Reward: {total_reward:.2f}")
                    print(f"  Final position: {env.current_pos}")
                    print(f"  Distance to start: {np.linalg.norm(env.current_pos - env.start_pos):.2f}")
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted.")
        finally:
            env.close()
    
    # ===== SB3 AGENTS (DQN, PPO, SAC, A2C) =====
    else:
        # Wrap the environment if using SAC
        if AGENT_TYPE == "SAC":
            env = ContinuousToDiscreteWrapper(env)
        
        # Set model path
        if AGENT_TYPE == "DQN":
            model_path = f"Results/{ENV_TYPE}/DQN_results/trained_dqn"
        elif AGENT_TYPE == "PPO":
            model_path = f"Results/{ENV_TYPE}/PPO_results/trained_ppo"
        elif AGENT_TYPE == "SAC":
            model_path = f"Results/{ENV_TYPE}/SAC_results/trained_sac"
        elif AGENT_TYPE == "A2C":
            model_path = f"Results/{ENV_TYPE}/A2C_results/trained_a2c"
        else:
            print(f"✗ Unknown agent type: {AGENT_TYPE}")
            env.close()
            return
        
        # Load model
        try:
            if AGENT_TYPE == "DQN":
                model = DQN.load(model_path)
            elif AGENT_TYPE == "PPO":
                model = PPO.load(model_path)
            elif AGENT_TYPE == "SAC":
                model = SAC.load(model_path)
            elif AGENT_TYPE == "A2C":
                model = A2C.load(model_path)
            print(f"✓ Loaded {AGENT_TYPE} agent from {model_path}")
        except FileNotFoundError:
            print(f"✗ Error: Could not find model at {model_path}")
            env.close()
            return
        
        trajectory = [env.unwrapped.current_pos.copy()]
        total_reward = 0
        steps = 0
        
        try:
            while True:
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                trajectory.append(env.unwrapped.current_pos.copy())
                total_reward += reward
                steps += 1
                
                if SHOW_DIAGNOSTICS:
                    unwrapped = env.unwrapped
                    phase = "Midpoint" if unwrapped.phase == 0 else "Return"
                    dist_to_target = np.linalg.norm(
                        unwrapped.current_pos - (unwrapped.midpoint if unwrapped.phase == 0 else unwrapped.start_pos)
                    )
                    print(f"Step {steps}: Phase={phase} | Pos={unwrapped.current_pos} | Dist={dist_to_target:.2f} | Reward={reward:+.3f}")
                
                env.render()
                time.sleep(0.25)
                
                if terminated or truncated:
                    unwrapped = env.unwrapped
                    print(f"\n✓ Episode finished!\n  Steps: {steps} | Reward: {total_reward:.2f}")
                    print(f"  Final position: {unwrapped.current_pos}")
                    print(f"  Distance to start: {np.linalg.norm(unwrapped.current_pos - unwrapped.start_pos):.2f}")
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted.")
        finally:
            env.close()

if __name__ == "__main__":
    run_test()
