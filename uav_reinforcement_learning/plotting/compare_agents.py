import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================
# COMPARISON PLOT TOOL
# ============================

def get_available_agents(env_type):
    """Get list of available agents for a given environment."""
    results_dir = f"Results/{env_type}"
    if not os.path.exists(results_dir):
        return []
    
    agents = []
    for folder in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, folder)):
            agent_name = folder.replace("_results", "").upper()
            agents.append(agent_name)
    
    return sorted(agents)


def select_environment():
    """Let user select environment type."""
    envs = ["original env", "improved env"]
    print("\nAvailable environments:")
    for i, env in enumerate(envs, 1):
        print(f"  {i}. {env}")
    
    while True:
        try:
            choice = int(input("Select environment (1-2): "))
            if 1 <= choice <= len(envs):
                return envs[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


def select_num_agents(available_agents):
    """Let user select how many agents to compare."""
    max_agents = len(available_agents)
    print(f"\nHow many agents would you like to compare? (2-{max_agents})")
    
    while True:
        try:
            num = int(input(f"Enter number of agents (2-{max_agents}): "))
            if 2 <= num <= max_agents:
                return num
            print(f"Invalid choice. Please enter a number between 2 and {max_agents}.")
        except ValueError:
            print("Please enter a number.")



def select_agents(env_type, num_agents=3):
    """Let user select agents to compare."""
    agents = get_available_agents(env_type)
    
    if len(agents) < num_agents:
        print(f"\nError: Only {len(agents)} agent(s) available. Need at least {num_agents}.")
        return None
    
    print(f"\nAvailable agents for '{env_type}':")
    for i, agent in enumerate(agents, 1):
        print(f"  {i}. {agent}")
    
    selected = []
    for agent_num in range(1, num_agents + 1):
        while True:
            try:
                choice = int(input(f"Select agent {agent_num} (1-{len(agents)}): "))
                if 1 <= choice <= len(agents):
                    agent = agents[choice - 1]
                    if agent not in selected:
                        selected.append(agent)
                        break
                    else:
                        print("That agent was already selected. Choose a different one.")
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    return selected


def load_training_data(env_type, agent_type):
    """Load training data for an agent."""
    agent_name = agent_type.lower()
    results_dir = f"Results/{env_type}"
    
    # Find the actual folder name (case-insensitive search)
    try:
        folders = os.listdir(results_dir)
        agent_folder = None
        for folder in folders:
            if folder.lower() == f"{agent_name}_results":
                agent_folder = folder
                break
        
        if not agent_folder:
            print(f"Warning: Folder not found for {agent_type} in {results_dir}")
            return None
        
        filename = os.path.join(results_dir, agent_folder, f"training_history_{agent_name}.npz")
        
        data = np.load(filename)
        sum_rates = data['sum_rates'].flatten()
        return sum_rates
    except FileNotFoundError as e:
        print(f"Warning: File not found for {agent_type} at {filename}")
        return None
    except Exception as e:
        print(f"Error loading {agent_type}: {e}")
        return None


def calculate_moving_average(data, window):
    """Calculate moving average with valid window."""
    if len(data) < window:
        window = max(1, len(data) // 2)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid'), window


def plot_comparison(env_type, agents):
    """Create a comprehensive comparison plot for multiple agents."""
    
    print(f"\nLoading data for {', '.join(agents)}...")
    
    # Load data for all agents
    agent_data = {}
    for agent in agents:
        data = load_training_data(env_type, agent)
        if data is None:
            print(f"Error: Could not load data for {agent}.")
            return
        agent_data[agent] = data
    
    # Define colors dynamically for any number of agents
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = {}
    for i, agent in enumerate(agents):
        colors[agent] = color_palette[i % len(color_palette)]
    
    # Create figure with subplots
    num_agents = len(agents)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ========== Plot 1: Overlaid training curves (top, spanning both columns) ==========
    ax_main = fig.add_subplot(gs[0, :])
    
    for agent in agents:
        data = agent_data[agent]
        episodes = np.arange(len(data))
        window = max(10, min(100, len(data) // 10))
        
        # Raw traces
        ax_main.plot(episodes, data, alpha=0.08, color=colors[agent], linewidth=0.5)
        
        # Moving averages
        ma, w = calculate_moving_average(data, window)
        x = np.arange(w - 1, len(data))
        ax_main.plot(x, ma, color=colors[agent], linewidth=2.5, 
                     label=f"{agent} (n={len(data)}, MA window={w})")
    
    ax_main.set_xlabel('Episode', fontsize=12)
    ax_main.set_ylabel('Avg Sum-Rate per Step', fontsize=12)
    ax_main.set_title(f'Learning Curves Comparison - {env_type}', fontsize=14, fontweight='bold')
    ax_main.legend(fontsize=11, loc='best')
    ax_main.grid(True, alpha=0.3)
    
    # ========== Plot 2-4: Individual agent curves ==========
    for idx, agent in enumerate(agents):
        ax = fig.add_subplot(gs[1, idx if idx < 2 else 1])
        data = agent_data[agent]
        episodes = np.arange(len(data))
        window = max(10, min(100, len(data) // 10))
        
        ax.plot(episodes, data, alpha=0.15, color=colors[agent], linewidth=0.5)
        
        ma, w = calculate_moving_average(data, window)
        x = np.arange(w - 1, len(data))
        ax.plot(x, ma, color=colors[agent], linewidth=2.5, label=agent)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Avg Sum-Rate per Step', fontsize=11)
        ax.set_title(f'{agent} Training Curve', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Remove the 4th subplot if only 3 agents
    if num_agents == 3:
        fig.delaxes(fig.axes[3])
    
    agents_str = '_vs_'.join(agents)
    plt.savefig(f'comparison_{agents_str}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to 'comparison_{agents_str}.png'")
    plt.show()
    
    # ========== Print statistics ==========
    print("\n" + "=" * 80)
    print(f"COMPARISON SUMMARY: {' vs '.join(agents)} ({env_type})")
    print("=" * 80)
    
    last_100_avgs = {}
    for agent in agents:
        data = agent_data[agent]
        n = len(data)
        first_n = data[:min(100, n)]
        last_n = data[-min(100, n):]
        last_100_avgs[agent] = np.mean(last_n)
        
        print(f"\n{agent}:")
        print(f"  Total episodes      : {n}")
        print(f"  Min sum-rate        : {np.min(data):.4f}")
        print(f"  Max sum-rate        : {np.max(data):.4f}")
        print(f"  Mean sum-rate       : {np.mean(data):.4f}")
        print(f"  Std dev             : {np.std(data):.4f}")
        print(f"  First 100 avg       : {np.mean(first_n):.4f}")
        print(f"  Last 100 avg        : {np.mean(last_n):.4f}")
        print(f"  Training progress   : {np.mean(last_n) - np.mean(first_n):+.4f}")
    
    # Relative comparison
    print(f"\n{'Relative Performance (Last 100 episodes):'}")
    for agent in agents:
        print(f"  {agent}: {last_100_avgs[agent]:.4f}")
    
    best_agent = max(last_100_avgs, key=last_100_avgs.get)
    second_agent = sorted(last_100_avgs.items(), key=lambda x: x[1], reverse=True)[1] if num_agents > 1 else (best_agent, 0)
    
    best_val = last_100_avgs[best_agent]
    second_val = second_agent[1]
    diff = best_val - second_val
    pct_diff = (diff / best_val) * 100 if best_val != 0 else 0
    
    print(f"\n  🏆 Best: {best_agent} with {best_val:.4f}")
    if num_agents > 1:
        print(f"  Runner-up: {second_agent[0]} with {second_val:.4f} ({diff:.4f} behind, {pct_diff:.2f}%)")
    
    print("=" * 80)



def main():
    """Main execution."""
    print("\n╔════════════════════════════════════════╗")
    print("║   AGENT COMPARISON TOOL                ║")
    print("╚════════════════════════════════════════╝")
    
    env_type = select_environment()
    print(f"\nSelected environment: {env_type}")
    
    available_agents = get_available_agents(env_type)
    num_agents = select_num_agents(available_agents)
    
    agents = select_agents(env_type, num_agents=num_agents)
    if agents is None:
        return
    
    print(f"\nSelected agents: {', '.join(agents)}")
    
    plot_comparison(env_type, agents)


if __name__ == "__main__":
    main()
