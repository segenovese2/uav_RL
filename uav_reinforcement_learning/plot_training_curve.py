import numpy as np
import matplotlib.pyplot as plt

# ============================
# SELECT AGENTS TO PLOT
# ============================
AGENT_TYPES = ["QLEARNING"]  # Options: 'QLEARNING', 'DQN', 'PPO', 'A2C'
# ============================


def plot_training_curves(agent_types):
    """Plot average sum-rate per step learning curves from saved training history."""

    print("Loading training history...")

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    results_found = False

    for i, agent_type in enumerate(agent_types):
        filename = f"training_history_{agent_type.lower()}.npz"

        try:
            data = np.load(filename)
            sum_rates = data['sum_rates'].flatten()  # ensure 1-D

            if len(sum_rates) == 0:
                print(f"  {agent_type}: file loaded but array is empty, skipping.")
                continue

            print(f"  {agent_type}: {len(sum_rates)} episodes loaded  "
                  f"(min={sum_rates.min():.4f}, max={sum_rates.max():.4f}, "
                  f"mean={sum_rates.mean():.4f})")

            results_found = True
            color = colors[i % len(colors)]

            episodes = np.arange(len(sum_rates))

            # raw trace, very transparent
            ax.plot(episodes, sum_rates, alpha=0.15, color=color, linewidth=0.5)

            # moving average -- window is at most 10% of data, at least 10
            window = max(10, min(100, len(sum_rates) // 10))
            if len(sum_rates) >= window:
                kernel = np.ones(window) / window
                moving_avg = np.convolve(sum_rates, kernel, mode='valid')
                # x axis: centre the window so the line aligns with episodes
                x_avg = np.arange(window - 1, len(sum_rates))
                ax.plot(
                    x_avg, moving_avg,
                    color=color, linewidth=2.5,
                    label=f"{agent_type} (n={len(sum_rates)}, window={window})"
                )
            else:
                ax.plot(
                    episodes, sum_rates,
                    color=color, linewidth=2,
                    label=f"{agent_type} (n={len(sum_rates)})"
                )

        except FileNotFoundError:
            print(f"  {agent_type}: file not found -- {filename}")
        except Exception as e:
            print(f"  {agent_type}: error loading -- {e}")

    if not results_found:
        print("\nNo training history files found. Train an agent first.")
        plt.close()
        return

    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Avg Sum-Rate per Step', fontsize=12)
    ax.set_title('Learning Curves: Avg Sum-Rate per Step over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_filename = 'training_learning_curves.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {plot_filename}")

    plt.show()

    # summary table
    print("\n" + "=" * 70)
    print("LEARNING CURVE SUMMARY")
    print("=" * 70)

    for agent_type in agent_types:
        filename = f"training_history_{agent_type.lower()}.npz"
        try:
            data = np.load(filename)
            sum_rates = data['sum_rates'].flatten()
            if len(sum_rates) == 0:
                continue

            n = len(sum_rates)
            first_n = sum_rates[:min(100, n)]
            last_n = sum_rates[-min(100, n):]

            print(f"\n{agent_type}:")
            print(f"  Episodes trained : {n}")
            print(f"  First 100 avg    : {np.mean(first_n):.4f}")
            print(f"  Last  100 avg    : {np.mean(last_n):.4f}")
            print(f"  Improvement      : {np.mean(last_n) - np.mean(first_n):.4f}")
            print(f"  Best episode     : {np.max(sum_rates):.4f}")
        except Exception:
            pass

    print("=" * 70)


if __name__ == "__main__":
    plot_training_curves(AGENT_TYPES)
