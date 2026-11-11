import sys

import matplotlib.pyplot as plt
import pandas as pd


def plot_rewards(log_file, gamma=0.99):
    columns = [
        "episode",
        "epsilon",
        "episode_reward",
        "rolling_average",
        "average_loss",
        "average_q_value",
    ]

    data = pd.read_csv(
        log_file,
        skiprows=1,
        names=columns,
    )

    plt.plot(
        data["episode"],
        data["rolling_average"],
        label="Average Reward",
    )

    plt.axhline(
        y=gamma,
        color="red",
        linestyle="--",
        label=f"Gamma ({gamma})",
    )

    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("DQN Average Reward and Gamma")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python plotting.py <log_file> [gamma]")
        sys.exit(1)

    log_file = sys.argv[1]
    gamma = float(sys.argv[2]) if len(sys.argv) == 3 else 0.99

    plot_rewards(log_file, gamma)