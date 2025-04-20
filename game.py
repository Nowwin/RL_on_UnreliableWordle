import random
import math  # at the top of the file

# Define a basic Wordle environment
class WordleEnv:
    def __init__(self, word_list, max_attempts=12, deception_prob=0.0):
        self.word_list = word_list
        self.max_attempts = max_attempts
        self.deception_prob = deception_prob  # probability of corrupting non-green feedback per letter
        self.reset()

    def reset(self):
        self.solution = random.choice(self.word_list)
        self.attempts = 0
        self.history = []  # Stores tuples of (guess, feedback)
        return self.history

    # New helper: change deception probability on the fly.
    def set_deception_prob(self, deception_prob):
        self.deception_prob = deception_prob

    def step(self, guess):
        self.attempts += 1
        feedback = self._get_feedback(guess)
        self.history.append((guess, feedback))
        done = False

        # Count feedback
        num_green = sum(1 for f in feedback if f == 'green')
        num_yellow = sum(1 for f in feedback if f == 'yellow')

        # Base shaped reward
        partial_reward = num_green * 0.5 + num_yellow * 0.2
        step_penalty = -0.1  # Discourage long games
        reward = partial_reward + step_penalty

        # Win reward scaled by attempts
        if guess == self.solution:
            done = True
            efficiency_bonus = max(0, (self.max_attempts - self.attempts)) * 0.5
            reward = 10.0 + efficiency_bonus  # strong incentive for early win
        elif self.attempts >= self.max_attempts:
            done = True

        return self.history, reward, done, {}



    # def step(self, guess):
    #     self.attempts += 1
    #     feedback = self._get_feedback(guess)
    #     self.history.append((guess, feedback))
    #     done = False

    #     # Exponential reward per letter
    #     reward = 0.0
    #     for fb in feedback:
    #         if fb == 'green':
    #             reward += math.exp(1)       
    #         elif fb == 'yellow':
    #             reward += math.exp(0.5)     
    #         else:  # gray
    #             reward -= math.exp(1)       

    #     # Bonus for solving
    #     if guess == self.solution:
    #         reward += 10.0  # small bonus on top
    #         done = True
    #     elif self.attempts >= self.max_attempts:
    #         done = True

    #     return self.history, reward, done, {}


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

        # Introduce unreliability for non-green feedback only:
        for i in range(len(feedback)):
            if feedback[i] != 'green' and random.random() < self.deception_prob:
                # Flip "yellow" to "gray" and vice versa.
                feedback[i] = 'gray' if feedback[i] == 'yellow' else 'yellow'
                
        return feedback

    def render(self):
        print("\nGame History:")
        for guess, feedback in self.history:
            print(f"Guess: {guess} -> Feedback: {feedback}")
        print("\n")
