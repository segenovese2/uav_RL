import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================
# ENVIRONMENT COMPARISON TOOL
# ============================

def get_available_agents():
    """Get list of available agents across both environments."""
    agents = set()
    for env in ["original env", "improved env"]:
        results_dir = f"Results/{env}"
        if os.path.exists(results_dir):
            for folder in os.listdir(results_dir):
                if os.path.isdir(os.path.join(results_dir, folder)):
                    agent_name = folder.replace("_results", "").upper()
                    agents.add(agent_name)
    
    return sorted(agents)


def select_agent():
    """Let user select an agent to compare across environments."""
    agents = get_available_agents()
    
    if len(agents) == 0:
        print("Error: No agents found in any environment.")
        return None
    
    print("\nAvailable agents:")
    for i, agent in enumerate(agents, 1):
        print(f"  {i}. {agent}")
    
    while True:
        try:
            choice = int(input(f"Select agent (1-{len(agents)}): "))
            if 1 <= choice <= len(agents):
                return agents[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


def load_training_data(env_type, agent_type):
    """Load training data for an agent in a specific environment."""
    agent_name = agent_type.lower()
    results_dir = f"Results/{env_type}"
    
    try:
        folders = os.listdir(results_dir)
        agent_folder = None
        for folder in folders:
            if folder.lower() == f"{agent_name}_results":
                agent_folder = folder
                break
        
        if not agent_folder:
            return None
        
        filename = os.path.join(results_dir, agent_folder, f"training_history_{agent_name}.npz")
        
        data = np.load(filename)
        sum_rates = data['sum_rates'].flatten()
        return sum_rates
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {agent_type} from {env_type}: {e}")
        return None


def calculate_moving_average(data, window):
    """Calculate moving average with valid window."""
    if len(data) < window:
        window = max(1, len(data) // 2)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid'), window


def plot_env_comparison(agent):
    """Create comparison plot for an agent across environments."""
    
    print(f"\nLoading data for {agent} in both environments...")
    
    data_original = load_training_data("original env", agent)
    data_improved = load_training_data("improved env", agent)
    
    if data_original is None and data_improved is None:
        print(f"Error: Could not load data for {agent} in any environment.")
        return
    
    if data_original is None:
        print(f"Warning: {agent} not found in original environment.")
    if data_improved is None:
        print(f"Warning: {agent} not found in improved environment.")
    
    # Define colors
    colors = {
        "original env": '#1f77b4',  # blue
        "improved env": '#ff7f0e',  # orange
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ========== Plot 1: Overlaid training curves ==========
    ax_main = fig.add_subplot(gs[0, :])
    
    if data_original is not None:
        episodes_orig = np.arange(len(data_original))
        window_orig = max(10, min(100, len(data_original) // 10))
        
        # Raw trace
        ax_main.plot(episodes_orig, data_original, alpha=0.08, color=colors["original env"], linewidth=0.5)
        
        # Moving average
        ma_orig, w_orig = calculate_moving_average(data_original, window_orig)
        x_orig = np.arange(w_orig - 1, len(data_original))
        ax_main.plot(x_orig, ma_orig, color=colors["original env"], linewidth=2.5,
                     label=f"Original Env (n={len(data_original)}, MA window={w_orig})")
    
    if data_improved is not None:
        episodes_imp = np.arange(len(data_improved))
        window_imp = max(10, min(100, len(data_improved) // 10))
        
        # Raw trace
        ax_main.plot(episodes_imp, data_improved, alpha=0.08, color=colors["improved env"], linewidth=0.5)
        
        # Moving average
        ma_imp, w_imp = calculate_moving_average(data_improved, window_imp)
        x_imp = np.arange(w_imp - 1, len(data_improved))
        ax_main.plot(x_imp, ma_imp, color=colors["improved env"], linewidth=2.5,
                     label=f"Improved Env (n={len(data_improved)}, MA window={w_imp})")
    
    ax_main.set_xlabel('Episode', fontsize=12)
    ax_main.set_ylabel('Avg Sum-Rate per Step', fontsize=12)
    ax_main.set_title(f'{agent} Performance: Original vs Improved Environment', fontsize=14, fontweight='bold')
    ax_main.legend(fontsize=11, loc='best')
    ax_main.grid(True, alpha=0.3)
    
    # ========== Plot 2: Original Environment ==========
    if data_original is not None:
        ax_orig = fig.add_subplot(gs[1, 0])
        ax_orig.plot(episodes_orig, data_original, alpha=0.15, color=colors["original env"], linewidth=0.5)
        ax_orig.plot(x_orig, ma_orig, color=colors["original env"], linewidth=2.5, label="Original Env")
        ax_orig.set_xlabel('Episode', fontsize=11)
        ax_orig.set_ylabel('Avg Sum-Rate per Step', fontsize=11)
        ax_orig.set_title(f'{agent} - Original Environment', fontsize=12, fontweight='bold')
        ax_orig.legend(fontsize=10)
        ax_orig.grid(True, alpha=0.3)
    
    # ========== Plot 3: Improved Environment ==========
    if data_improved is not None:
        ax_imp = fig.add_subplot(gs[1, 1])
        ax_imp.plot(episodes_imp, data_improved, alpha=0.15, color=colors["improved env"], linewidth=0.5)
        ax_imp.plot(x_imp, ma_imp, color=colors["improved env"], linewidth=2.5, label="Improved Env")
        ax_imp.set_xlabel('Episode', fontsize=11)
        ax_imp.set_ylabel('Avg Sum-Rate per Step', fontsize=11)
        ax_imp.set_title(f'{agent} - Improved Environment', fontsize=12, fontweight='bold')
        ax_imp.legend(fontsize=10)
        ax_imp.grid(True, alpha=0.3)
    
    plt.savefig(f'comparison_{agent}_environments.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to 'comparison_{agent}_environments.png'")
    plt.show()
    
    # ========== Print statistics ==========
    print("\n" + "=" * 90)
    print(f"ENVIRONMENT COMPARISON: {agent}")
    print("=" * 90)
    
    if data_original is not None:
        n = len(data_original)
        first_n = data_original[:min(100, n)]
        last_n = data_original[-min(100, n):]
        
        print(f"\nORIGINAL ENVIRONMENT:")
        print(f"  Total episodes      : {n}")
        print(f"  Min sum-rate        : {np.min(data_original):.4f}")
        print(f"  Max sum-rate        : {np.max(data_original):.4f}")
        print(f"  Mean sum-rate       : {np.mean(data_original):.4f}")
        print(f"  Std dev             : {np.std(data_original):.4f}")
        print(f"  First 100 avg       : {np.mean(first_n):.4f}")
        print(f"  Last 100 avg        : {np.mean(last_n):.4f}")
        print(f"  Training progress   : {np.mean(last_n) - np.mean(first_n):+.4f}")
    
    if data_improved is not None:
        n = len(data_improved)
        first_n = data_improved[:min(100, n)]
        last_n = data_improved[-min(100, n):]
        
        print(f"\nIMPROVED ENVIRONMENT:")
        print(f"  Total episodes      : {n}")
        print(f"  Min sum-rate        : {np.min(data_improved):.4f}")
        print(f"  Max sum-rate        : {np.max(data_improved):.4f}")
        print(f"  Mean sum-rate       : {np.mean(data_improved):.4f}")
        print(f"  Std dev             : {np.std(data_improved):.4f}")
        print(f"  First 100 avg       : {np.mean(first_n):.4f}")
        print(f"  Last 100 avg        : {np.mean(last_n):.4f}")
        print(f"  Training progress   : {np.mean(last_n) - np.mean(first_n):+.4f}")
    
    # Relative comparison
    if data_original is not None and data_improved is not None:
        print(f"\nENVIRONMENT IMPACT (Last 100 episodes):")
        orig_last = np.mean(data_original[-min(100, len(data_original)):])
        imp_last = np.mean(data_improved[-min(100, len(data_improved)):])
        
        print(f"  Original Env avg    : {orig_last:.4f}")
        print(f"  Improved Env avg    : {imp_last:.4f}")
        
        diff = imp_last - orig_last
        pct_diff = (diff / orig_last) * 100 if orig_last != 0 else 0
        
        if diff > 0:
            print(f"  ✓ Improvement: +{diff:.4f} ({pct_diff:.2f}%)")
        else:
            print(f"  ✗ Degradation: {diff:.4f} ({pct_diff:.2f}%)")
    
    print("=" * 90)


def main():
    """Main execution."""
    print("\n╔════════════════════════════════════════╗")
    print("║   ENVIRONMENT COMPARISON TOOL          ║")
    print("║   Original vs Improved                 ║")
    print("╚════════════════════════════════════════╝")
    
    agent = select_agent()
    if agent is None:
        return
    
    print(f"\nSelected agent: {agent}")
    
    plot_env_comparison(agent)


if __name__ == "__main__":
    main()
