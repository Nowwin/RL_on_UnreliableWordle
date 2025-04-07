import random

# Define a basic Wordle environment
class WordleEnv:
    def __init__(self, word_list, max_attempts=6):
        self.word_list = word_list
        self.max_attempts = max_attempts
        self.reset()

    def reset(self):
        self.solution = random.choice(self.word_list)
        self.attempts = 0
        self.history = []  # Stores tuples of (guess, feedback)
        return self.history

    def step(self, guess):
        self.attempts += 1
        feedback = self._get_feedback(guess)
        self.history.append((guess, feedback))
        done = False
        reward = -0.1  # Small penalty per guess
        
        if guess == self.solution:
            reward = 1.0  # Reward for guessing correctly
            done = True
        elif self.attempts >= self.max_attempts:
            done = True

        return self.history, reward, done, {}

    def _get_feedback(self, guess):
        # Initialize all feedback as "gray"
        feedback = ['gray'] * len(guess)
        solution_chars = list(self.solution)
        
        # First pass: mark correct letters as "green"
        for i in range(len(guess)):
            if guess[i] == solution_chars[i]:
                feedback[i] = 'green'
                solution_chars[i] = None  # Mark as used
        
        # Second pass: mark misplaced letters as "yellow"
        for i in range(len(guess)):
            if feedback[i] != 'green' and guess[i] in solution_chars:
                feedback[i] = 'yellow'
                solution_chars[solution_chars.index(guess[i])] = None
        
        return feedback

    def render(self):
        print("\nGame History:")
        for guess, feedback in self.history:
            print(f"Guess: {guess} -> Feedback: {feedback}")
        print("\n")