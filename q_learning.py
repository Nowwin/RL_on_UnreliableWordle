import random
import matplotlib.pyplot as plt
from game import WordleEnv

def load_word_list(filename):
    """Load a list of 5-letter words from a text file."""
    with open(filename, 'r') as f:
        return [line.strip().lower() for line in f if len(line.strip()) == 5]

def get_state(env):
    """
    Represent the current state of the environment.
    For the initial state (no guess yet) return "start".
    Afterwards, return a tuple: (last guess, feedback_str)
    Where feedback_str is a 5-character string with:
      G for green, Y for yellow, X for gray.
    """
    if env.attempts == 0 or not env.history:
        return "start"
    else:
        guess, feedback = env.history[-1]
        fb_str = "".join("G" if f=="green" else "Y" if f=="yellow" else "X" for f in feedback)
        return (guess, fb_str)

def epsilon_greedy(Q, state, word_list, epsilon):
    """
    Choose an action based on epsilon-greedy policy.
    If state not in Q, then initialize with 0 for all actions.
    """
    if state not in Q:
        Q[state] = {word: 0.0 for word in word_list}
    if random.random() < epsilon:
        return random.choice(word_list)
    # Otherwise, choose action with maximum Q-value.
    q_vals = Q[state]
    max_q = max(q_vals.values())
    # If there are ties, choose randomly among the best.
    best_actions = [a for a, q in q_vals.items() if q == max_q]
    return random.choice(best_actions)

def initialize_Q_for_state(Q, state, word_list):
    """Ensure that a Q entry exists for the state."""
    if state not in Q:
        Q[state] = {word: 0.0 for word in word_list}

def q_learning_episode(word_list, env, Q, alpha=0.1, gamma=1.0, epsilon=0.2):
    """
    Runs one episode using the Q-learning algorithm.
    Returns number of attempts, total reward, and a win flag.
    The Q table is updated in-place.
    """
    env.reset()
    state = get_state(env)
    initialize_Q_for_state(Q, state, word_list)
    total_reward = 0
    attempts = 0
    done = False

    while not done:
        attempts += 1
        
        # Choose an action via epsilon-greedy
        action = epsilon_greedy(Q, state, word_list, epsilon)
        
        # Take action in environment.
        history, reward, done, _ = env.step(action)
        total_reward += reward
        
        next_state = get_state(env)
        initialize_Q_for_state(Q, next_state, word_list)
        
        # Q-learning update.
        max_next_q = max(Q[next_state].values()) if Q[next_state] else 0.0
        current_q = Q[state][action]
        Q[state][action] = current_q + alpha * (reward + gamma * max_next_q - current_q)
        
        state = next_state

    win = True if reward > 0 else False
    return attempts, total_reward, win

def run_q_learning_episodes(word_list, num_episodes, deception_prob, alpha=0.1, gamma=1.0, epsilon=0.2):
    """
    Runs q-learning for a given number of episodes and deception probability.
    Returns win rate, list of attempts, and list of total rewards.
    """
    win_count = 0
    attempts_list = []
    rewards_list = []
    Q = {}  # Initialize the Q-table as a dictionary

    for _ in range(num_episodes):
        env = WordleEnv(word_list, deception_prob=deception_prob)
        attempts, total_reward, win = q_learning_episode(word_list, env, Q, alpha, gamma, epsilon)
        win_count += int(win)
        attempts_list.append(attempts)
        rewards_list.append(total_reward)
        
    win_rate = win_count / num_episodes
    return win_rate, attempts_list, rewards_list

if __name__ == "__main__":
    word_list = load_word_list("wordList.txt")
    num_episodes = 10000
    deception_levels = [0.0, 0.05, 0.1]

    win_rates = {}
    avg_attempts = {}
    avg_rewards = {}

    for d in deception_levels:
        print(f"Running {num_episodes} episodes with deception probability = {d}")
        win_rate, attempts, rewards = run_q_learning_episodes(word_list, num_episodes, deception_prob=d,
                                                              alpha=0.1, gamma=1.0, epsilon=0.2)
        win_rates[d] = win_rate
        avg_attempts[d] = sum(attempts) / len(attempts)
        avg_rewards[d] = sum(rewards) / len(rewards)
        print(f"Deception Prob: {d} | Win Rate: {win_rate*100:.2f}%, Avg Attempts: {avg_attempts[d]:.2f}, Avg Reward: {avg_rewards[d]:.2f}\n")

    # Plotting
    plt.figure(figsize=(12, 4))

    # Subplot 1: Win Rate
    plt.subplot(1, 3, 1)
    probs = [str(d) for d in deception_levels]
    rates = [win_rates[d] for d in deception_levels]
    plt.bar(probs, rates, color='skyblue')
    plt.xlabel("Deception Probability")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs. Deception Probability")
    plt.ylim(0, 1)

    # Subplot 2: Average Attempts
    plt.subplot(1, 3, 2)
    attempts_vals = [avg_attempts[d] for d in deception_levels]
    plt.bar(probs, attempts_vals, color='salmon')
    plt.xlabel("Deception Probability")
    plt.ylabel("Avg Attempts")
    plt.title("Avg Attempts vs. Deception Probability")

    # Subplot 3: Average Reward
    plt.subplot(1, 3, 3)
    rewards_vals = [avg_rewards[d] for d in deception_levels]
    plt.bar(probs, rewards_vals, color='lightgreen')
    plt.xlabel("Deception Probability")
    plt.ylabel("Avg Reward")
    plt.title("Avg Reward vs. Deception Probability")

    plt.tight_layout()
    plt.show()
