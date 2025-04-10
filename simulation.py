import random
import matplotlib.pyplot as plt
from game import WordleEnv

def load_word_list(filename):
    """Load a list of 5-letter words from a text file."""
    with open(filename, 'r') as f:
        # Each line should contain one 5-letter word.
        return [line.strip().lower() for line in f if len(line.strip()) == 5]

def compute_feedback(solution, guess):
    """
    Compute feedback for a given guess compared to the solution.
    Follows the same logic as the WordleEnv._get_feedback method.
    """
    feedback = ['gray'] * len(guess)
    solution_chars = list(solution)
    # First pass: mark correct letters as green.
    for i in range(len(guess)):
        if guess[i] == solution_chars[i]:
            feedback[i] = 'green'
            solution_chars[i] = None  # Mark as used.
    # Second pass: mark letters that exist but in the wrong spot as yellow.
    for i in range(len(guess)):
        if feedback[i] != 'green' and guess[i] in solution_chars:
            feedback[i] = 'yellow'
            solution_chars[solution_chars.index(guess[i])] = None
    return feedback

def is_consistent(candidate, guess, feedback):
    """
    Check if a candidate word would have produced the given feedback
    when compared with the guess.
    """
    return compute_feedback(candidate, guess) == feedback

def simulation_agent(word_list, env, verbose=False):
    """
    A candidate-elimination agent that rolls out one episode.
    Returns:
        attempts: number of guesses taken,
        total_reward: cumulative reward in the episode,
        win: True if the word was guessed correctly.
    """
    candidates = word_list.copy()
    env.reset()
    attempt = 0
    total_reward = 0
    done = False

    while not done:
        attempt += 1
        # Choose a guess from the candidate set (here, a random candidate).
        guess = random.choice(candidates)
        if verbose:
            print(f"Attempt {attempt}: Guessing '{guess}'")
        history, reward, done, _ = env.step(guess)
        total_reward += reward
        
        if verbose:
            env.render()
        
        if done:
            win = (reward > 0)
            if verbose:
                if win:
                    print("Agent successfully guessed the word!")
                else:
                    print(f"Agent failed. The word was: '{env.solution}'")
            return attempt, total_reward, win

        # Update the candidate list based on the feedback of the last guess.
        last_feedback = history[-1][1]
        new_candidates = [w for w in candidates if is_consistent(w, guess, last_feedback)]
        if new_candidates:
            candidates = new_candidates
        else:
            # If filtering leaves no candidates, reset to the full list.
            if verbose:
                print("No candidates left after filtering. Resetting candidate set.")
            candidates = word_list.copy()

def run_episodes(word_list, num_episodes, deception_prob):
    """
    Runs a set of episodes with a given deception probability.
    Returns win_rate, list of attempts, and list of total rewards.
    """
    win_count = 0
    attempts_list = []
    rewards_list = []
    
    for _ in range(num_episodes):
        env = WordleEnv(word_list, deception_prob=deception_prob)
        attempts, total_reward, win = simulation_agent(word_list, env, verbose=False)
        win_count += int(win)
        attempts_list.append(attempts)
        rewards_list.append(total_reward)
    
    win_rate = win_count / num_episodes
    return win_rate, attempts_list, rewards_list

if __name__ == "__main__":
    # Load word list from file (ensure "wordList.txt" is in the same directory)
    word_list = load_word_list("wordList2.txt")
    num_episodes = 10000
    deception_levels = [0.0, 0.05, 0.1]

    # Dictionaries to hold results for each deception probability.
    win_rates = {}
    avg_attempts = {}
    avg_rewards = {}
    
    for d in deception_levels:
        print(f"Running {num_episodes} episodes with deception probability = {d}")
        win_rate, attempts, rewards = run_episodes(word_list, num_episodes, deception_prob=d)
        win_rates[d] = win_rate
        avg_attempts[d] = sum(attempts) / len(attempts)
        avg_rewards[d] = sum(rewards) / len(rewards)
        print(f"Deception Prob: {d} | Win Rate: {win_rate*100:.2f}%, Avg Attempts: {avg_attempts[d]:.2f}, Avg Reward: {avg_rewards[d]:.2f}\n")
    
    # Plotting the win rate for each deception level.
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
