import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load your training stats
with open("train_stats_deception_0.1.pkl", "rb") as f:
    stats = pickle.load(f)

# Extract data
losses = stats['losses']
rewards = stats['rewards']
win_rates = stats['win_rates']

# Smooth function (optional, to make plots nicer)
def smooth(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Plot
plt.figure(figsize=(18, 5))

# Reward plot
plt.subplot(1, 3, 1)
plt.plot(smooth(rewards), label='Smoothed Reward')
plt.title("Smoothed Rewards over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

# Win rate plot
plt.subplot(1, 3, 2)
plt.plot(smooth(win_rates), label='Smoothed Win Rate')
plt.title("Smoothed Win Rate over Episodes")
plt.xlabel("Episode")
plt.ylabel("Win Rate")
plt.grid(True)
plt.legend()

# Loss plot
plt.subplot(1, 3, 3)
plt.plot(smooth(losses), label='Smoothed Loss', color='orange')
plt.title("Smoothed Loss over Updates")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
