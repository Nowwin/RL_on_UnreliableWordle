import random
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

def simulation_agent(word_list, env):
    """
    A simple simulation agent that uses candidate elimination to guess the solution.
    It starts with the entire word list as candidates and, after each guess, filters out
    candidates that do not match the feedback received.
    """
    candidates = word_list.copy()
    history = env.reset()
    done = False
    attempt = 0

    while not done:
        attempt += 1
        # Choose a guess from the candidate set (here, a random candidate).
        guess = random.choice(candidates)
        print(f"Attempt {attempt}: Guessing '{guess}'")
        history, reward, done, _ = env.step(guess)
        env.render()
        
        if done:
            if reward > 0:
                print("Agent successfully guessed the word!")
            else:
                print(f"Agent failed to guess. The word was: '{env.solution}'")
            break

        # Update the candidate list based on the feedback of the last guess.
        last_feedback = history[-1][1]
        new_candidates = [w for w in candidates if is_consistent(w, guess, last_feedback)]
        if new_candidates:
            candidates = new_candidates
        else:
            # If filtering leaves no candidates, reset to the full list.
            print("No candidates left after filtering. Resetting candidate set.")
            candidates = word_list.copy()

if __name__ == "__main__":
    # Load the comprehensive word list from wordList.txt.
    word_list = load_word_list("wordList.txt")
    # Initialize the Wordle environment from game.py.
    env = WordleEnv(word_list)
    # Run the simulation with our candidate-elimination agent.
    simulation_agent(word_list, env)
