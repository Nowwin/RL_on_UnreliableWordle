import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def smooth(data, window_size=100):
    """Apply a moving average to smooth the data"""
    if len(data) < window_size:
        return data
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # Pad the beginning to maintain the same length
    padding = np.full(window_size-1, smoothed[0])
    return np.concatenate([padding, smoothed])

def save_results(results, filename):
    """Save experiment results to a file"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filename):
    """Load experiment results from a file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_training_curves(results, deception_levels, save_path=None):
    """
    Plot training curves for different metrics across deception levels
    
    Args:
        results: Dictionary containing results for each deception level
        deception_levels: List of deception probabilities
        save_path: Path to save the figure
    """
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    plt.figure(figsize=(18, 10))
    
    # Plot win rates during training
    plt.subplot(2, 2, 1)
    for i, d in enumerate(deception_levels):
        win_rates = results[d]['train']['win_rates']
        plt.plot(win_rates, label=f"Deception={d}", color=colors[i%len(colors)])
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot rewards during training
    plt.subplot(2, 2, 2)
    for i, d in enumerate(deception_levels):
        rewards = results[d]['train']['rewards']
        smoothed_rewards = smooth(rewards)
        plt.plot(smoothed_rewards, label=f"Deception={d}", color=colors[i%len(colors)])
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Smoothed Rewards During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot losses during training
    plt.subplot(2, 2, 3)
    for i, d in enumerate(deception_levels):
        if 'losses' in results[d]['train'] and len(results[d]['train']['losses']) > 0:
            losses = results[d]['train']['losses']
            smoothed_losses = smooth(losses)
            plt.plot(smoothed_losses, label=f"Deception={d}", color=colors[i%len(colors)])
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Smoothed Loss During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bar chart of final metrics
    plt.subplot(2, 2, 4)
    x = np.arange(len(deception_levels))
    width = 0.25
    
    win_rates = [results[d]['eval']['win_rate'] for d in deception_levels]
    avg_rewards = [results[d]['eval']['avg_reward'] for d in deception_levels]
    avg_attempts = [results[d]['eval']['avg_attempts']/6 for d in deception_levels]  # Normalize to [0,1]
    
    plt.bar(x - width, win_rates, width, label='Win Rate', color='skyblue')
    plt.bar(x, avg_rewards, width, label='Avg Reward (normalized)', color='salmon')
    plt.bar(x + width, avg_attempts, width, label='Avg Attempts (normalized)', color='lightgreen')
    
    plt.xlabel("Deception Probability")
    plt.xticks(x, [str(d) for d in deception_levels])
    plt.ylabel("Value")
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def compare_with_simulation(dqn_results, simulation_results, deception_levels, save_path=None):
    """
    Compare DQN results with simulation results
    
    Args:
        dqn_results: Dictionary containing DQN results for each deception level
        simulation_results: Dictionary containing simulation results for each deception level
        deception_levels: List of deception probabilities
        save_path: Path to save the figure
    """
    plt.figure(figsize=(15, 5))
    
    # Plot win rates
    plt.subplot(1, 3, 1)
    x = np.arange(len(deception_levels))
    width = 0.35
    
    dqn_win_rates = [dqn_results[d]['eval']['win_rate'] for d in deception_levels]
    sim_win_rates = [simulation_results[d]['win_rate'] for d in deception_levels]
    
    plt.bar(x - width/2, dqn_win_rates, width, label='DQN', color='skyblue')
    plt.bar(x + width/2, sim_win_rates, width, label='Simulation', color='lightblue')
    
    plt.xlabel("Deception Probability")
    plt.xticks(x, [str(d) for d in deception_levels])
    plt.ylabel("Win Rate")
    plt.title("Win Rate Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot average attempts
    plt.subplot(1, 3, 2)
    dqn_avg_attempts = [dqn_results[d]['eval']['avg_attempts'] for d in deception_levels]
    sim_avg_attempts = [simulation_results[d]['avg_attempts'] for d in deception_levels]
    
    plt.bar(x - width/2, dqn_avg_attempts, width, label='DQN', color='salmon')
    plt.bar(x + width/2, sim_avg_attempts, width, label='Simulation', color='lightsalmon')
    
    plt.xlabel("Deception Probability")
    plt.xticks(x, [str(d) for d in deception_levels])
    plt.ylabel("Avg Attempts")
    plt.title("Avg Attempts Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot average rewards
    plt.subplot(1, 3, 3)
    dqn_avg_rewards = [dqn_results[d]['eval']['avg_reward'] for d in deception_levels]
    sim_avg_rewards = [simulation_results[d]['avg_reward'] for d in deception_levels]
    
    plt.bar(x - width/2, dqn_avg_rewards, width, label='DQN', color='lightgreen')
    plt.bar(x + width/2, sim_avg_rewards, width, label='Simulation', color='palegreen')
    
    plt.xlabel("Deception Probability")
    plt.xticks(x, [str(d) for d in deception_levels])
    plt.ylabel("Avg Reward")
    plt.title("Avg Reward Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def visualize_game_playthrough(agent, env, word_list, num_games=5):
    """
    Visualize a few game playthroughs using the trained agent
    
    Args:
        agent: Trained WordleAgent
        env: WordleEnv instance
        word_list: List of all possible words
        num_games: Number of games to visualize
    """
    from dqn import encode_history
    
    def render_colored_feedback(guess, feedback):
        """Render a guess with colored feedback"""
        colors = {
            'green': '\033[92m',  # Green
            'yellow': '\033[93m', # Yellow
            'gray': '\033[90m',   # Gray
            'end': '\033[0m'      # Reset
        }
        
        result = ""
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            result += f"{colors[fb]}{letter.upper()}{colors['end']}"
        return result
    
    for game in range(num_games):
        print(f"\n===== Game {game+1} =====")
        
        # Reset the environment
        history = env.reset()
        state = encode_history(history, word_list)
        done = False
        won = False
        
        print(f"Solution: {env.solution}")
        
        while not done:
            # Select the best action (no exploration)
            action_idx, action_word = agent.get_action(state, epsilon=0)
            
            # Take a step in the environment
            next_history, reward, done, _ = env.step(action_word)
            next_state = encode_history(next_history, word_list)
            
            # Get the latest guess and feedback
            guess, feedback = next_history[-1]
            
            # Display the guess with feedback
            colored_guess = render_colored_feedback(guess, feedback)
            print(f"Guess {len(next_history)}: {colored_guess}")
            
            # Move to the next state
            state = next_state
            
            if done and reward > 0:
                won = True
        
        if won:
            print(f"Agent WON in {len(next_history)} attempts!")
        else:
            print(f"Agent LOST. The solution was: {env.solution}")
        
        print("="*20)