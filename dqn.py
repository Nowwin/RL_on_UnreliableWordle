import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import deque, namedtuple, Counter
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from game import WordleEnv

# Define Experience tuple for storing game steps
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    """Prioritized Experience replay buffer for better sample efficiency"""
    def __init__(self, capacity, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
    
    def push(self, experience, priority=None):
        """Store an experience in the buffer"""
        if priority is None:
            # If no priority given, use max priority to ensure this gets sampled
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """Sample a batch of experiences based on priorities"""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Return sampled experiences and their indices for priority updates
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for the given indices"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class WordleDQN(nn.Module):
    """Improved Deep Q-Network for Wordle with larger capacity"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(WordleDQN, self).__init__()
        
        # Larger network with more layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights for better gradient flow
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Apply batch norm only during training (when batch size > 1)
        if x.shape[0] > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        
        return self.fc4(x)

def encode_history(history, word_list, letter_freq=None):
    """
    Enhanced state representation for Wordle
    
    Args:
        history: List of (guess, feedback) tuples
        word_list: List of all possible words
        letter_freq: Letter frequency dictionary for the word list
    
    Returns:
        A fixed-length state vector with more features
    """
    # Initialize letter frequencies if not provided
    if letter_freq is None:
        letter_freq = {}
        all_letters = ''.join(word_list)
        total_letters = len(all_letters)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            letter_freq[letter] = all_letters.count(letter) / total_letters
    
    # Features we'll encode:
    # 1. Count of each feedback type (green, yellow, gray) for each position (15 features)
    # 2. Count of attempted words (1 feature)
    # 3. Information about letters:
    #    - Known green letters at each position (26*5 features)
    #    - Known yellow letters at each position (26*5 features)
    #    - Known gray letters (26 features)
    #    - Letter frequency weighting for unused letters (26 features)
    # 4. Number of remaining attempts (1 feature)
    
    # Initialize features
    position_feedback = [[0, 0, 0] for _ in range(5)]  # [green, yellow, gray] for each position
    green_letters = [[0] * 26 for _ in range(5)]  # Position-specific green letters
    yellow_letters = [[0] * 26 for _ in range(5)]  # Position-specific yellow letters
    gray_letters = [0] * 26  # Letters that aren't in the word
    
    # Process history
    for guess, feedback in history:
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            letter_idx = ord(letter) - ord('a')
            
            if fb == 'green':
                position_feedback[i][0] += 1
                green_letters[i][letter_idx] = 1
            elif fb == 'yellow':
                position_feedback[i][1] += 1
                yellow_letters[i][letter_idx] = 1
            else:  # gray
                position_feedback[i][2] += 1
                
                # Only mark as gray if not green or yellow elsewhere
                is_used = False
                for j in range(5):
                    if j != i and j < len(guess) and guess[j] == letter:
                        if feedback[j] in ['green', 'yellow']:
                            is_used = True
                            break
                
                if not is_used:
                    gray_letters[letter_idx] = 1
    
    # Calculate letter frequency weighting for unused letters
    letter_weights = []
    for i in range(26):
        letter = chr(i + ord('a'))
        if gray_letters[i] == 0 and all(green_letters[j][i] == 0 for j in range(5)) and all(yellow_letters[j][i] == 0 for j in range(5)):
            # This letter is unused, weight by frequency
            letter_weights.append(letter_freq.get(letter, 0.0))
        else:
            letter_weights.append(0.0)
    
    # Construct the state vector
    state = []
    
    # Flatten position feedback
    for pos in position_feedback:
        state.extend(pos)
    
    # Add number of attempts and remaining attempts
    max_attempts = 6  # Assuming 6 attempts max
    state.append(len(history))
    state.append(max_attempts - len(history))
    
    # Flatten green and yellow letter indicators
    for pos in green_letters:
        state.extend(pos)
    
    for pos in yellow_letters:
        state.extend(pos)
    
    # Add gray letters and letter weights
    state.extend(gray_letters)
    state.extend(letter_weights)
    
    return np.array(state, dtype=np.float32)

class WordleAgent:
    """Improved agent that plays Wordle using a DQN with better exploration strategies"""
    def __init__(self, state_dim, action_dim, word_list, hidden_dim=256, lr=0.0005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_list = word_list
        
        # Calculate letter frequencies for the word list (used in state encoding)
        all_letters = ''.join(word_list)
        total_letters = len(all_letters)
        self.letter_freq = {}
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            self.letter_freq[letter] = all_letters.count(letter) / total_letters
        
        # Initialize Q networks
        self.q_network = WordleDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = WordleDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Use Adam optimizer with smaller learning rate for stability
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=500, verbose=True
        )
        
        # Use Huber loss for better handling of outliers
        self.loss_fn = nn.SmoothL1Loss()
        
        # Use prioritized replay for better sample efficiency
        self.replay_buffer = PrioritizedReplayBuffer(20000)
        
        # Word to index mapping
        self.word_to_idx = {word: i for i, word in enumerate(word_list)}
        self.idx_to_word = {i: word for i, word in enumerate(word_list)}
    
    def get_action(self, state, epsilon, history=None):
        """
        Enhanced action selection with better exploration
        
        Args:
            state: Current state
            epsilon: Exploration rate
            history: Game history for smarter exploration
        
        Returns:
            Selected action index and corresponding word
        """
        if random.random() < epsilon:
            # Smart exploration: try to pick words that haven't been tried
            # and that use letters we haven't seen yet
            if history and len(history) > 0:
                # Get previously tried words
                tried_words = [guess for guess, _ in history]
                
                # Get used letters
                used_letters = set()
                for word in tried_words:
                    used_letters.update(set(word))
                
                # Calculate scores for each word based on new letters
                word_scores = []
                for word in self.word_list:
                    if word in tried_words:
                        # Don't repeat words
                        continue
                    
                    # Count new letters in this word
                    new_letters = set(word) - used_letters
                    score = len(new_letters)
                    
                    # Add some randomness to avoid getting stuck
                    score += random.random()
                    
                    word_scores.append((word, score))
                
                if word_scores:
                    # Pick words with highest scores (most new letters)
                    word_scores.sort(key=lambda x: x[1], reverse=True)
                    top_words = word_scores[:max(1, int(len(word_scores) * 0.1))]  # Top 10%
                    chosen_word = random.choice(top_words)[0]
                    return self.word_to_idx[chosen_word], chosen_word
            
            # Fallback to random selection
            action_idx = random.randrange(len(self.word_list))
        else:
            # Use the Q-network for exploitation
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        return action_idx, self.idx_to_word[action_idx]
    
    def update(self, batch_size, gamma=0.99, beta=0.4):
        """Update the Q-network using a batch of prioritized experiences"""
        if len(self.replay_buffer) < batch_size:
            return 0
        
        # Sample a batch of experiences with priorities
        experiences, indices = self.replay_buffer.sample(batch_size)
        
        # Unpack experiences
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([exp.done for exp in experiences]).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values with Double Q-learning
        with torch.no_grad():
            # Get actions from current network
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # Get Q-values from target network
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            # Zero out Q-values for terminal states
            next_q_values = next_q_values * (1 - dones)
        
        # Compute target Q values
        target_q_values = rewards + gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Compute TD errors for prioritization
        with torch.no_grad():
            td_errors = torch.abs(target_q_values - current_q_values).cpu().numpy()
        
        # Update priorities in the replay buffer
        new_priorities = td_errors + 1e-6  # Add small constant to avoid zero priority
        self.replay_buffer.update_priorities(indices, new_priorities)
        
        # Update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self, tau=1.0):
        """Update the target network using soft update or hard update"""
        if tau == 1.0:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft update
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, path):
        """Save the model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

def train_dqn_agent(agent, env, num_episodes, batch_size=128, gamma=0.99, 
                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99954, 
                   target_update=10, save_path=None, verbose=False,
                   reward_threshold=0.5, early_stopping_threshold=500):
    """
    Train the DQN agent with improved training parameters and early stopping
    """
    rewards = []
    win_rates = []
    losses = []
    epsilon = epsilon_start
    win_count = 0
    episodes_without_improvement = 0
    best_win_rate = 0
    tau = 0.01  # Soft update parameter
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset the environment
        history = env.reset()
        state = encode_history(history, agent.word_list, agent.letter_freq)
        total_reward = 0
        done = False
        
        # Play one episode
        while not done:
            # Select an action with smart exploration
            action_idx, action_word = agent.get_action(state, epsilon, history)
            
            # Take a step in the environment
            next_history, reward, done, _ = env.step(action_word)
            next_state = encode_history(next_history, agent.word_list, agent.letter_freq)
            
            # Store the experience
            agent.replay_buffer.push(Experience(state, action_idx, reward, next_state, done))
            
            # Update the Q-network multiple times for better learning
            for _ in range(4):  # Update network multiple times per step
                loss = agent.update(batch_size, gamma)
                if loss > 0:
                    losses.append(loss)
            
            # Move to the next state
            state = next_state
            history = next_history
            total_reward += reward
        
        # Soft update the target network every step
        agent.update_target_network(tau)
        
        # Hard update the target network periodically
        if episode % target_update == 0:
            agent.update_target_network(1.0)
        
        # Update epsilon with a slower decay schedule
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Record episode statistics
        rewards.append(total_reward)
        win_count += (total_reward > 0)
        current_win_rate = win_count / (episode + 1)
        win_rates.append(current_win_rate)
        
        # Update learning rate based on performance
        if episode % 100 == 0:
            agent.scheduler.step(current_win_rate)
        
        # Check for early stopping
        if current_win_rate > best_win_rate:
            best_win_rate = current_win_rate
            episodes_without_improvement = 0
            if save_path:
                agent.save(f"{save_path}_best")
        else:
            episodes_without_improvement += 1
        
        # Print verbose output periodically
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}: Reward = {total_reward}, Win Rate = {current_win_rate:.4f}, Epsilon = {epsilon:.4f}")
            
            # Print examples of recent wins/losses for debugging
            recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
            recent_win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            print(f"Recent 100 episodes win rate: {recent_win_rate:.4f}")
        
        # Early stopping if win rate threshold reached or no improvement for too long
        if current_win_rate >= reward_threshold or episodes_without_improvement >= early_stopping_threshold:
            if current_win_rate >= reward_threshold:
                print(f"Early stopping: win rate threshold {reward_threshold} reached!")
            else:
                print(f"Early stopping: no improvement for {early_stopping_threshold} episodes.")
            break
    
    # Save the final trained model
    if save_path:
        agent.save(save_path)
    
    return {
        'rewards': rewards,
        'win_rates': win_rates,
        'losses': losses,
        'final_win_rate': win_rates[-1],
        'best_win_rate': best_win_rate,
        'episodes_trained': episode + 1
    }

def evaluate_agent(agent, env, num_episodes, render_every=0):
    """
    Evaluate the trained agent with optional rendering
    """
    rewards = []
    win_count = 0
    attempts = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluation"):
        # Reset the environment
        history = env.reset()
        state = encode_history(history, agent.word_list, agent.letter_freq)
        total_reward = 0
        done = False
        episode_attempts = 0
        
        if render_every > 0 and episode % render_every == 0:
            print(f"\nEvaluation game {episode}, target word: {env.solution}")
        
        # Play one episode
        while not done:
            # Select the best action (no exploration)
            action_idx, action_word = agent.get_action(state, epsilon=0)
            
            if render_every > 0 and episode % render_every == 0:
                print(f"Attempt {episode_attempts+1}: Agent guesses '{action_word}'")
            
            # Take a step in the environment
            next_history, reward, done, _ = env.step(action_word)
            next_state = encode_history(next_history, agent.word_list, agent.letter_freq)
            
            if render_every > 0 and episode % render_every == 0:
                _, feedback = next_history[-1]
                print(f"Feedback: {feedback}")
            
            # Move to the next state
            state = next_state
            total_reward += reward
            episode_attempts += 1
            
            if done:
                if reward > 0:
                    win_count += 1
                    if render_every > 0 and episode % render_every == 0:
                        print(f"Agent WON in {episode_attempts} attempts!")
                elif render_every > 0 and episode % render_every == 0:
                    print(f"Agent LOST. The solution was: {env.solution}")
        
        rewards.append(total_reward)
        attempts.append(episode_attempts)
    
    win_rate = win_count / num_episodes
    avg_reward = sum(rewards) / num_episodes
    avg_attempts = sum(attempts) / num_episodes
    
    # Calculate attempt distribution for winners
    winning_attempts = [a for a, r in zip(attempts, rewards) if r > 0]
    attempt_distribution = Counter(winning_attempts)
    
    return {
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'avg_attempts': avg_attempts,
        'attempt_distribution': attempt_distribution
    }

def run_dqn_experiment(word_list, num_episodes=5000, eval_episodes=1000, 
                      deception_levels=[0.0, 0.05, 0.1], batch_size=128,
                      hidden_dim=256, reward_threshold=0.5):
    """
    Run experiments with different deception levels using improved parameters
    """
    # Calculate enhanced state dimensions
    state_dim = (5 * 3) + 2 + (26 * 5 * 2) + 26 + 26
    action_dim = len(word_list)
    
    results = {}
    
    for deception_prob in deception_levels:
        print(f"\n=== Running experiment with deception probability {deception_prob} ===")
        
        # Create the environment and agent
        env = WordleEnv(word_list, max_attempts=6, deception_prob=deception_prob)
        agent = WordleAgent(state_dim, action_dim, word_list, hidden_dim=hidden_dim, lr=0.0005)
        
        # Train the agent with improved parameters
        save_path = f"wordle_dqn_deception_{deception_prob}.pt"
        train_stats = train_dqn_agent(
            agent, env, num_episodes=num_episodes, 
            batch_size=batch_size, gamma=0.99, 
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
            target_update=10, save_path=save_path, verbose=True,
            reward_threshold=reward_threshold
        )
        
        # Evaluate the trained agent
        eval_stats = evaluate_agent(agent, env, num_episodes=eval_episodes, render_every=100)
        
        results[deception_prob] = {
            'train': train_stats,
            'eval': eval_stats
        }
        
        # Print detailed results
        print(f"Results for deception probability {deception_prob}:")
        print(f"  Win Rate: {eval_stats['win_rate']:.4f}")
        print(f"  Avg Reward: {eval_stats['avg_reward']:.4f}")
        print(f"  Avg Attempts: {eval_stats['avg_attempts']:.4f}")
        
        if eval_stats['attempt_distribution']:
            print("  Attempt distribution for winning games:")
            for attempt, count in sorted(eval_stats['attempt_distribution'].items()):
                print(f"    {attempt} attempts: {count} games ({count/sum(eval_stats['attempt_distribution'].values())*100:.1f}%)")
    
    return results

def plot_detailed_results(results, deception_levels):
    """
    Plot more detailed results of the experiments
    """
    plt.figure(figsize=(20, 12))
    
    # Plot win rates
    plt.subplot(2, 3, 1)
    win_rates = [results[d]['eval']['win_rate'] for d in deception_levels]
    plt.bar([str(d) for d in deception_levels], win_rates, color='skyblue')
    plt.xlabel("Deception Probability")
    plt.ylabel("Win Rate")
    plt.title("DQN: Win Rate vs. Deception Probability")
    plt.ylim(0, 1)
    
    # Plot average attempts
    plt.subplot(2, 3, 2)
    avg_attempts = [results[d]['eval']['avg_attempts'] for d in deception_levels]
    plt.bar([str(d) for d in deception_levels], avg_attempts, color='salmon')
    plt.xlabel("Deception Probability")
    plt.ylabel("Avg Attempts")
    plt.title("DQN: Avg Attempts vs. Deception Probability")
    
    # Plot average rewards
    plt.subplot(2, 3, 3)
    avg_rewards = [results[d]['eval']['avg_reward'] for d in deception_levels]
    plt.bar([str(d) for d in deception_levels], avg_rewards, color='lightgreen')
    plt.xlabel("Deception Probability")
    plt.ylabel("Avg Reward")
    plt.title("DQN: Avg Reward vs. Deception Probability")
    
    # Plot training curves
    plt.subplot(2, 3, 4)
    for i, d in enumerate(deception_levels):
        win_rates = results[d]['train']['win_rates']
        episodes = min(len(win_rates), 1000)  # Show at most 1000 episodes for clarity
        plt.plot(win_rates[:episodes], label=f"Deception={d}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Win Rate")
    plt.title("Training Win Rate Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot smoothed reward during training
    plt.subplot(2, 3, 5)
    for i, d in enumerate(deception_levels):
        rewards = results[d]['train']['rewards']
        episodes = min(len(rewards), 1000)  # Show at most 1000 episodes
        
        # Apply smoothing
        window_size = 50
        smoothed = np.convolve(rewards[:episodes], np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed, label=f"Deception={d}")
    
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Episode Reward")
    plt.title("Training Reward Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot attempt distribution for wins
    plt.subplot(2, 3, 6)
    for i, d in enumerate(deception_levels):
        if 'attempt_distribution' in results[d]['eval']:
            dist = results[d]['eval']['attempt_distribution']
            if dist:
                attempts = list(range(1, 7))
                counts = [dist.get(a, 0) for a in attempts]
                total = sum(counts)
                
                # Convert to percentages if we have wins
                if total > 0:
                    percentages = [count/total*100 for count in counts]
                    plt.bar([str(a) for a in attempts], percentages, alpha=0.7, label=f"Deception={d}")
    
    plt.xlabel("Number of Attempts")
    plt.ylabel("Percentage of Wins")
    plt.title("Attempt Distribution for Winning Games")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("dqn_detailed_results.png")
    plt.show()

def load_word_list(filename):
    """Load a list of 5-letter words from a text file."""
    with open(filename, 'r') as f:
        return [line.strip().lower() for line in f if len(line.strip()) == 5]

if __name__ == "__main__":
    # Load the word list
    word_list = load_word_list("wordList2.txt")
    
    # Print some statistics
    print(f"Loaded {len(word_list)} words")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the experiments with improved parameters
    deception_levels = [0.0, 0.05, 0.1]
    results = run_dqn_experiment(
        word_list, 
        num_episodes=10000,  # More episodes for better learning
        eval_episodes=1000,
        deception_levels=deception_levels,
        batch_size=128,
        hidden_dim=256,
        reward_threshold=0.4  # Stop early if win rate reaches 40%
    )
    
    # Plot detailed results
    plot_detailed_results(results, deception_levels)