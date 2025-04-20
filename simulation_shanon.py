import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from game import WordleEnv

def load_word_list(filename):
    with open(filename, 'r') as f:
        return [line.strip().lower() for line in f if len(line.strip()) == 5]

def compute_feedback(solution, guess):
    feedback = ['gray'] * len(guess)
    solution_chars = list(solution)
    for i in range(len(guess)):
        if guess[i] == solution_chars[i]:
            feedback[i] = 'green'
            solution_chars[i] = None
    for i in range(len(guess)):
        if feedback[i] != 'green' and guess[i] in solution_chars:
            feedback[i] = 'yellow'
            solution_chars[solution_chars.index(guess[i])] = None
    return feedback

def is_consistent(candidate, guess, feedback):
    return compute_feedback(candidate, guess) == feedback

def feedback_to_key(feedback):
    return ''.join(f[0] for f in feedback)

def compute_expected_entropy(candidates, guess):
    counts = defaultdict(int)
    for sol in candidates:
        fb = compute_feedback(sol, guess)
        counts[feedback_to_key(fb)] += 1
    N = len(candidates)
    entropy = 0.0
    for count in counts.values():
        p = count / N
        entropy -= p * math.log2(p)
    return entropy

def shannon_agent(word_list, env):
    candidates = word_list.copy()
    history = []
    env.reset()
    done = False
    win = False
    while not done:
        # Filter by past feedback
        for guess, fb in history:
            candidates = [w for w in candidates if is_consistent(w, guess, fb)]
        # Choose guess minimizing expected entropy
        best_guess = min(candidates, key=lambda g: compute_expected_entropy(candidates, g))
        history, reward, done, _ = env.step(best_guess)
        if best_guess == env.solution:
            win = True
    return win

def evaluate_shannon(word_list, num_episodes, deception_levels):
    win_rates = {}
    for d in deception_levels:
        print(f"Deception Level: {d}")
        wins = 0
        for episode in range(1, num_episodes + 1):
            print(f"Episode: {episode}")
            env = WordleEnv(word_list, deception_prob=d)
            if shannon_agent(word_list, env):
                wins += 1
        win_rates[d] = wins / num_episodes
    return win_rates


if __name__ == "__main__":
    word_list = load_word_list("wordList2.txt")
    num_episodes = 500
    deception_levels = [0.0, 0.1, 0.2]
    win_rates = evaluate_shannon(word_list, num_episodes, deception_levels)
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar([str(d) for d in deception_levels], [win_rates[d] for d in deception_levels])
    plt.xlabel("Deception Probability")
    plt.ylabel("Win Rate")
    plt.title(f"Shannon-Agent Win Rate over {num_episodes} Episodes")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
