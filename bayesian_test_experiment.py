import numpy as np
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
import pickle
from game import WordleEnv

# --- Load Word List Function ---
def load_word_list(filename):
    words = []
    try:
        with open(filename, 'r') as f:
            words = [line.strip().lower() for line in f if len(line.strip()) == 5]
    except FileNotFoundError:
         print(f"Error: File not found at {filename}")
    return words

class BayesianWordleAgent:
    """
    A Bayesian RL agent for playing Wordle under uncertainty with trust modeling.
    Uses Thompson sampling and posterior belief updates with adaptive trust parameters.
    Includes separate handling for gray letters with confidence values.
    """
    def __init__(self, word_list, alpha=1.0, beta=1.0):
        self.word_list = word_list
        self.full_word_set = set(word_list)
        self.possible_words = self.word_list.copy()

        # Bayesian priors: Beta(alpha, beta)
        self.letter_position_prior = {}
        for position in range(5):
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                self.letter_position_prior[(letter, position)] = (alpha, beta)

        self.letter_in_word_prior = {}
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            self.letter_in_word_prior[letter] = (alpha, beta)

        # Game state trackers
        self.not_in_word = set()         # Traditional set but used less aggressively
        self.gray_letters = set()        # Track all gray letters separately
        self.gray_confidence = {}        # Store confidence for each gray letter
        self.letter_inconsistency = defaultdict(int)  # Track inconsistencies per letter
        self.letter_feedback_history = defaultdict(list)  # {letter: [(position, feedback), ...]}
        
        self.green_letters = {}          # {position: letter} confirmed correct position
        self.yellow_letters = defaultdict(set)  # {position: set(letters)} - letters present but NOT at this pos
        self.known_present_letters = set()  # Letters confirmed present (green or yellow)

        # Trust parameters for different feedback types - higher initial values
        self.trust_green = 1.0
        self.trust_yellow = 0.9
        self.trust_gray = 0.9
        
        # Define a trust floor to prevent values from dropping too low
        self.trust_floor = 0.1
        
        # Track feedback consistency for trust modeling
        self.feedback_consistency = {}
        
        # Exploration-exploitation balance
        self.exploration_weight = 0.4
        self.exploration_cap = 0.6

        # Confidence threshold for word filtering
        self.confidence_threshold = 0.4
        
        # Keep track of past guesses to avoid repeating
        self.past_guesses = []
        
        # Keep track of rejected words that might need reconsideration
        self.rejected_words = set()
        
        # Counter for consecutive repeated guesses - to trigger reset
        self.repeat_guess_count = 0
        
        # Time to try something new threshold
        self.max_repeat_count = 2
        
        # Current game attempt number
        self.current_attempt = 0
        
        # Force diversity in guesses
        self.letter_diversity_bonus = 0.3

    def update_beliefs(self, guess, feedback):
        """ Update belief model based on guess and feedback with trust weighting """
        # Track this guess to avoid repetition
        self.past_guesses.append(guess)
        self.current_attempt += 1
        
        # Check if we just repeated a guess
        if len(self.past_guesses) >= 2 and self.past_guesses[-1] == self.past_guesses[-2]:
            self.repeat_guess_count += 1
        else:
            self.repeat_guess_count = 0
        
        prev_not_in_word = self.not_in_word.copy()
        prev_green_letters = self.green_letters.copy()
        prev_yellow_letters = self.yellow_letters.copy()
        
        current_guess_greens = {}
        current_guess_yellows = defaultdict(list)
        current_guess_grays = []

        # Track feedback for consistency analysis
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            # Track per letter-position
            if (letter, i) not in self.feedback_consistency:
                self.feedback_consistency[(letter, i)] = []
            self.feedback_consistency[(letter, i)].append(fb)
            
            # Track per letter across positions
            self.letter_feedback_history[letter].append((i, fb))

        # First pass: Update beliefs based on feedback with trust weighting
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            # Determine the weight based on trust parameters
            weight = self.trust_green if fb == "green" else self.trust_yellow if fb == "yellow" else self.trust_gray
            
            if fb == "green":
                self.green_letters[i] = letter
                self.known_present_letters.add(letter)
                if letter in self.not_in_word: 
                    self.not_in_word.remove(letter)
                if letter in self.gray_letters:
                    self.gray_letters.remove(letter)
                current_guess_greens[i] = letter

                alpha_pos, beta_pos = self.letter_position_prior[(letter, i)]
                self.letter_position_prior[(letter, i)] = (alpha_pos + weight, beta_pos)

                alpha_in, beta_in = self.letter_in_word_prior[letter]
                self.letter_in_word_prior[letter] = (alpha_in + weight, beta_in)

            elif fb == "yellow":
                self.yellow_letters[i].add(letter)
                self.known_present_letters.add(letter)
                if letter in self.not_in_word: 
                    self.not_in_word.remove(letter)
                if letter in self.gray_letters:
                    self.gray_letters.remove(letter)
                current_guess_yellows[letter].append(i)

                alpha_pos, beta_pos = self.letter_position_prior[(letter, i)]
                self.letter_position_prior[(letter, i)] = (alpha_pos, beta_pos + weight)

                alpha_in, beta_in = self.letter_in_word_prior[letter]
                self.letter_in_word_prior[letter] = (alpha_in + weight, beta_in)

            else:  # Gray
                current_guess_grays.append(letter)
                alpha_pos, beta_pos = self.letter_position_prior[(letter, i)]
                self.letter_position_prior[(letter, i)] = (alpha_pos, beta_pos + weight)

                is_truly_absent = True
                for j, (other_letter, other_fb) in enumerate(zip(guess, feedback)):
                    if other_letter == letter and other_fb in ["green", "yellow"]:
                        is_truly_absent = False
                        break

                if is_truly_absent:
                    # Add to gray letters set but with confidence tracking
                    if letter not in self.known_present_letters:
                        self.gray_letters.add(letter)
                        # Initialize/update confidence for this gray letter
                        if letter not in self.gray_confidence:
                            self.gray_confidence[letter] = self.trust_gray
                        # Regular priors update
                        alpha_in, beta_in = self.letter_in_word_prior[letter]
                        self.letter_in_word_prior[letter] = (alpha_in, beta_in + weight)
                    
                    # Only add to not_in_word if we have high confidence it's truly absent
                    if letter not in self.known_present_letters and self.gray_confidence.get(letter, 0) > 0.7:
                        self.not_in_word.add(letter)
        
        # Update trust parameters based on feedback consistency
        self._update_trust_parameters()
                     
        self._filter_possible_words()
        
        # If we're stuck in a loop, inject some randomness
        if self.repeat_guess_count >= self.max_repeat_count:
            self._reset_beliefs()
            # Log this for debugging
            print("RESETTING beliefs due to repeated guesses!")

    def _reset_beliefs(self):
        """Reset some beliefs to break out of loops"""
        # Loosen constraints on gray letters
        self.not_in_word = set()
        
        # Reduce confidence in gray letters
        for letter in self.gray_confidence:
            self.gray_confidence[letter] = self.trust_floor
        
        # Add some rejected words back to possible words
        if self.rejected_words:
            sample_size = min(50, len(self.rejected_words))
            if sample_size > 0:
                reconsider_words = random.sample(list(self.rejected_words), sample_size)
                self.possible_words.extend(reconsider_words)
        
        # If filtering has become too aggressive, relax constraints
        if len(self.possible_words) < 10:
            # Look for words that match green constraints but ignore other constraints
            green_only_matches = []
            for word in self.word_list:
                if all(word[pos] == letter for pos, letter in self.green_letters.items()):
                    green_only_matches.append(word)
            
            # Add some of these back if needed
            if green_only_matches:
                self.possible_words = list(set(self.possible_words + green_only_matches[:50]))
        
        # Reset repeat counter
        self.repeat_guess_count = 0

    def _update_trust_parameters(self):
        """
        Dynamically update trust parameters based on feedback consistency with a trust floor
        Also track letter-specific inconsistencies
        """
        # First: letter-position inconsistencies
        for (letter, position), feedbacks in self.feedback_consistency.items():
            if len(feedbacks) < 2:
                continue
                
            # Check for inconsistencies in feedback
            green_count = feedbacks.count("green")
            yellow_count = feedbacks.count("yellow")
            gray_count = feedbacks.count("gray")
            
            # Detect inconsistencies (e.g., same letter-position got different feedback)
            has_inconsistency = False
            if green_count > 0 and (yellow_count > 0 or gray_count > 0):
                has_inconsistency = True  # Green is inconsistent with yellow/gray
            if yellow_count > 0 and gray_count > 0:
                has_inconsistency = True  # Yellow is inconsistent with gray

            # If inconsistency detected, reduce trust in non-green feedback - but more gently
            if has_inconsistency:
                # Track letter-specific inconsistency
                self.letter_inconsistency[letter] += 1
                
                # Calculate letter-specific trust level
                letter_trust = max(self.trust_floor, self.trust_gray * (0.98 ** self.letter_inconsistency[letter]))
                if letter in self.gray_confidence:
                    self.gray_confidence[letter] = letter_trust
                
                # More aggressive reduction factor
                inconsistency_factor = 0.95
                self.trust_yellow = max(self.trust_floor, self.trust_yellow * inconsistency_factor)
                self.trust_gray = max(self.trust_floor, self.trust_gray * inconsistency_factor)
                
                # Increase exploration weight to compensate for uncertainty
                self.exploration_weight = min(self.exploration_cap, self.exploration_weight * 1.02)
                
                # Adjust confidence threshold
                self.confidence_threshold = max(0.2, self.confidence_threshold * 0.98)

        # Second: analyze each letter across positions
        for letter, history in self.letter_feedback_history.items():
            if len(history) < 2:
                continue
                
            feedbacks = [fb for _, fb in history]
            green_count = feedbacks.count("green")
            yellow_count = feedbacks.count("yellow")
            gray_count = feedbacks.count("gray")
            
            # Check for a letter being both present (green/yellow) and absent (gray)
            if (green_count > 0 or yellow_count > 0) and gray_count > 0:
                # Letter has inconsistent feedback - might be multiple occurrences or deception
                self.letter_inconsistency[letter] += 1
                
                # Calculate letter-specific trust
                letter_trust = max(self.trust_floor, self.trust_gray * (0.95 ** self.letter_inconsistency[letter]))
                if letter in self.gray_confidence:
                    self.gray_confidence[letter] = letter_trust
                
                # If letter is in gray_letters but has been seen as green/yellow elsewhere,
                # reduce confidence in it being truly absent
                if letter in self.gray_letters and letter in self.known_present_letters:
                    self.gray_confidence[letter] = max(self.trust_floor, self.gray_confidence[letter] * 0.9)
                    # Maybe remove from not_in_word if confidence too low
                    if letter in self.not_in_word and self.gray_confidence[letter] < 0.3:
                        self.not_in_word.remove(letter)

    def _filter_possible_words(self):
        """ Filter self.possible_words using a two-stage approach """
        if not self.possible_words: 
            self.possible_words = self.word_list.copy() 

        # Save original possible words to add to rejected set later
        original_possible = set(self.possible_words)

        # Stage 1: Apply only high-confidence constraints
        high_confidence_words = []
        for word in self.possible_words:
            if self._is_high_confidence_consistent(word):
                high_confidence_words.append(word)
        
        # Stage 2: If we have enough high-confidence words, use those
        if len(high_confidence_words) >= 5:
            new_possible_words = high_confidence_words
        else:
            # Regular filtering with more leniency
            new_possible_words = []
            for word in self.possible_words:
                if self._is_consistent_aggregated(word):
                    confidence = self._calculate_word_confidence(word)
                    if confidence >= self.confidence_threshold:
                        new_possible_words.append(word)
            
            # Further adjust if still filtering too much
            if len(new_possible_words) < 5 and len(self.possible_words) > 20:
                # Use lower threshold
                temp_threshold = self.confidence_threshold * 0.7
                new_possible_words = []
                for word in self.possible_words:
                    if self._is_consistent_aggregated(word):
                        confidence = self._calculate_word_confidence(word)
                        if confidence >= temp_threshold:
                            new_possible_words.append(word)
        
        # Add words that were filtered out to the rejected list
        self.rejected_words.update(original_possible - set(new_possible_words))
            
        # Ensure we always have some possible words
        if len(new_possible_words) >= 3:
            self.possible_words = new_possible_words
        elif self.possible_words:
            # Keep top 20% most confident words
            confidences = [(word, self._calculate_word_confidence(word)) 
                           for word in self.possible_words 
                           if self._is_consistent_aggregated(word)]
            confidences.sort(key=lambda x: x[1], reverse=True)
            top_n = max(5, int(len(self.possible_words) * 0.2))
            if confidences:
                self.possible_words = [word for word, _ in confidences[:top_n]]
        
        # Fallback: if all filtering failed
        if not self.possible_words:
            # Use full word list with minimal gray letter constraints
            full_list_candidates = []
            for word in self.word_list:
                if self._is_minimal_consistent(word):
                    full_list_candidates.append(word)
            
            if full_list_candidates:
                self.possible_words = full_list_candidates
            else:
                # Last resort: sample from full list
                self.possible_words = random.sample(self.word_list, min(100, len(self.word_list)))

    def _is_high_confidence_consistent(self, candidate):
        """Check if word is consistent with high-confidence constraints only"""
        candidate_letters = Counter(candidate)
        
        # Green checks (always high confidence)
        for pos, letter in self.green_letters.items():
            if candidate[pos] != letter: 
                return False
        
        # Yellow checks (always high confidence for letter presence)
        for letter in self.known_present_letters:
            if letter not in candidate_letters: 
                return False 
                
        # Yellow position checks (always high confidence)
        for pos, letters_at_pos in self.yellow_letters.items():
            for letter in letters_at_pos:
                if candidate[pos] == letter: 
                    return False 
                    
        # Only apply gray constraints for high-confidence gray letters
        for letter in self.gray_letters:
            if letter in candidate_letters and letter not in self.known_present_letters:
                # Only apply if high confidence
                if self.gray_confidence.get(letter, 0) > 0.7:
                    return False
                    
        return True

    def _is_consistent_aggregated(self, candidate):
        """
        Checks if a candidate word is consistent with known information,
        with probabilistic application of gray letter constraints
        """
        candidate_letters = Counter(candidate)
        
        # Green checks (must match)
        for pos, letter in self.green_letters.items():
            if candidate[pos] != letter: 
                return False
        
        # Yellow checks (letter must be present)
        for letter in self.known_present_letters:
            if letter not in candidate_letters: 
                return False 
                
        # Yellow position checks (letter must NOT be at this position)
        for pos, letters_at_pos in self.yellow_letters.items():
            for letter in letters_at_pos:
                if candidate[pos] == letter: 
                    return False 
                    
        # Gray checks with probabilistic application
        for letter in self.gray_letters:
            if letter in candidate_letters and letter not in self.known_present_letters:
                # Get confidence that this letter is truly absent
                confidence = self.gray_confidence.get(letter, self.trust_gray)
                
                # Apply constraint probabilistically based on confidence
                if random.random() < confidence:
                    return False  # Respect the gray feedback
                # Otherwise, ignore this gray letter constraint
                    
        return True

    def _is_minimal_consistent(self, candidate):
        """Apply only the most certain constraints - used as a fallback"""
        candidate_letters = Counter(candidate)
        
        # Green checks (must match)
        for pos, letter in self.green_letters.items():
            if candidate[pos] != letter: 
                return False
        
        # Yellow checks (letter must be present)
        for letter in self.known_present_letters:
            if letter not in candidate_letters: 
                return False 
                
        # Yellow position checks (letter must NOT be at this position)
        for pos, letters_at_pos in self.yellow_letters.items():
            for letter in letters_at_pos:
                if candidate[pos] == letter: 
                    return False 
                    
        # Only apply gray constraints for very high confidence gray letters
        for letter in self.not_in_word:  # Using not_in_word (stricter set) rather than gray_letters
            if letter in candidate_letters and letter not in self.known_present_letters:
                # Only apply if very high confidence
                if self.gray_confidence.get(letter, 0) > 0.8:
                    return False
                    
        return True

    def _calculate_word_confidence(self, word):
        """Calculate confidence score for a word based on current beliefs and trust parameters"""
        confidence = 1.0
        
        # Green letter confidence
        for pos, letter in self.green_letters.items():
            if word[pos] != letter:
                return 0.0  # Immediately rule out
            
            alpha, beta = self.letter_position_prior[(letter, pos)]
            letter_confidence = alpha / (alpha + beta) if alpha + beta > 0 else 0.5
            confidence *= letter_confidence * self.trust_green
        
        # Yellow letter confidence
        for letter in self.known_present_letters:
            if letter not in word:
                return 0.0  # Immediately rule out
            
            alpha, beta = self.letter_in_word_prior[letter]
            letter_confidence = alpha / (alpha + beta) if alpha + beta > 0 else 0.5
            confidence *= letter_confidence * self.trust_yellow
        
        # Yellow position constraints
        for pos, letters_at_pos in self.yellow_letters.items():
            for letter in letters_at_pos:
                if word[pos] == letter:
                    return 0.0  # Immediately rule out
                
                alpha, beta = self.letter_position_prior[(letter, pos)]
                letter_pos_confidence = beta / (alpha + beta) if alpha + beta > 0 else 0.5
                confidence *= letter_pos_confidence * self.trust_yellow
        
        # Gray letter confidence - more lenient for possible deception
        for letter in self.gray_letters:
            if letter in word and letter not in self.known_present_letters:
                # Use letter-specific confidence if available
                letter_confidence = self.gray_confidence.get(letter, self.trust_gray)
                
                # Apply partial penalty instead of immediately ruling out
                confidence *= (1.0 - letter_confidence * 0.7)  # Reduced from 0.8 for more leniency
                
                if confidence < 0.01:  # If confidence is very low, rule it out
                    return 0.0
        
        return confidence

    def _score_word(self, word, possible_letter_freq, num_possible):
        """ Score a word based on Bayesian beliefs, info gain, and entropy reduction. """
        # If this word has been guessed before, penalize heavily
        if word in self.past_guesses:
            return -1000.0  # Ensure we never repeat a guess!
            
        # Position score based on letter-position priors
        position_score = 0
        for i, letter in enumerate(word):
            alpha, beta = self.letter_position_prior[(letter, i)]
            # Use expected value instead of random sample for more stability
            position_prob = alpha / (alpha + beta) if alpha + beta > 0 else 0.5
            position_score += position_prob
            
        # Information gain score
        info_gain = 0
        if num_possible > 0:
            for letter in set(word):
                letter_count = possible_letter_freq.get(letter, 0)
                letter_freq_in_possible = letter_count / num_possible
                entropy_contribution = 1.0 - abs(0.5 - letter_freq_in_possible) * 2
                info_gain += entropy_contribution
                
        # Entropy reduction component for deceptive environments
        entropy_reduction = 0
        
        # Words that haven't been tried get a bonus
        if all(letter not in self.known_present_letters and letter not in self.gray_letters for letter in word):
            entropy_reduction += 0.5

        # Avoid words with many of the same letter when there's deception
        unique_letter_ratio = len(set(word)) / len(word)
        entropy_reduction += unique_letter_ratio * 0.5
        
        # Prefer words that test gray letters with low confidence
        gray_letter_test_bonus = 0
        for letter in self.gray_letters:
            if letter in word and self.gray_confidence.get(letter, 1.0) < 0.7:
                # Bonus for testing gray letters we're uncertain about
                gray_letter_test_bonus += (1.0 - self.gray_confidence.get(letter, 1.0)) * 0.3
        
        # Preference for common letters in English
        common_letters = "etaoinshrdlucmfwypvbgkjqxz"
        common_letter_bonus = sum(1.0/(common_letters.index(letter)+1) for letter in set(word) if letter in common_letters) / 10.0
        
        # Letter diversity bonus (prefer words with letters we haven't tried yet)
        letter_diversity = 0
        past_letters = set(''.join(self.past_guesses))
        for letter in set(word):
            if letter not in past_letters:
                letter_diversity += 0.2
        
        # Reconsideration bonus: if from rejected set, give small boost
        reconsideration_bonus = 0.1 if word in self.rejected_words else 0
                
        # Calculate final score with adaptive weights
        final_score = (position_score + 
                      self.exploration_weight * info_gain + 
                      0.3 * entropy_reduction + 
                      0.2 * common_letter_bonus +
                      gray_letter_test_bonus +
                      self.letter_diversity_bonus * letter_diversity +
                      reconsideration_bonus)
                      
        return final_score

    def select_action(self):
        """ Select the next word to guess using Thompson sampling & info gain. """
        # If very few possibilities, choose one directly - but avoid repeats!
        non_repeated_candidates = [w for w in self.possible_words if w not in self.past_guesses]
        
        if len(non_repeated_candidates) == 0:
            # All possible words have been guessed before!
            # If we're in later game, try one of the rejected words
            if self.current_attempt > 5 and self.rejected_words:
                # Try a random sample from rejected words
                candidates = list(self.rejected_words - set(self.past_guesses))
                if candidates:
                    return random.choice(candidates)
                    
            # Force diversity by picking a word with new letters
            past_letters = set(''.join(self.past_guesses))
            diverse_candidates = []
            for word in self.word_list:
                word_letters = set(word)
                # Count unique new letters
                new_letters = word_letters - past_letters
                if len(new_letters) >= 3 and word not in self.past_guesses:  # At least 3 new letters
                    diverse_candidates.append(word)
            
            if diverse_candidates:
                return random.choice(diverse_candidates)
            else:
                # Just pick any unused word as last resort
                unused_words = [w for w in self.word_list if w not in self.past_guesses]
                if unused_words:
                    return random.choice(unused_words)
                else:
                    # Truly stuck - pick random
                    return random.choice(self.word_list)
                    
        if len(non_repeated_candidates) == 1:
            return non_repeated_candidates[0]
        
        if len(non_repeated_candidates) <= 3:
            return random.choice(non_repeated_candidates)
            
        # Prepare letter frequency info
        possible_letter_freq = Counter()
        num_possible = len(self.possible_words)
        if num_possible > 0:
            for word in self.possible_words: 
                possible_letter_freq.update(word)
                
        # Score candidate words
        # Start with non-repeated candidates
        words_to_score = list(non_repeated_candidates)
        
        # Add some words from rejected list for consideration if we have few candidates
        if len(words_to_score) < 10 and self.rejected_words and self.current_attempt > 4:
            rejected_candidates = list(self.rejected_words - set(self.past_guesses))
            if rejected_candidates:
                sample_size = min(20, len(rejected_candidates))
                words_to_score.extend(random.sample(rejected_candidates, sample_size))
        
        # Add random words from full list for exploration if we have few candidates
        if len(words_to_score) < 10:
            # Get words with new letters we haven't tried
            past_letters = set(''.join(self.past_guesses))
            diverse_candidates = []
            for word in self.word_list:
                if word not in self.past_guesses and word not in words_to_score:
                    word_letters = set(word)
                    # Count unique new letters
                    new_letters = word_letters - past_letters
                    if len(new_letters) >= 3:  # At least 3 new letters
                        diverse_candidates.append(word)
            
            # Add some diverse candidates
            if diverse_candidates:
                sample_size = min(20, len(diverse_candidates))
                words_to_score.extend(random.sample(diverse_candidates, sample_size))
            else:
                # Add random other words
                exploration_candidates = [w for w in self.word_list if w not in self.past_guesses and w not in words_to_score]
                if exploration_candidates:
                    sample_size = min(20, len(exploration_candidates))
                    words_to_score.extend(random.sample(exploration_candidates, sample_size))
        
        word_scores = []
        for word in words_to_score:
            score = self._score_word(word, possible_letter_freq, num_possible)
            word_scores.append((word, score))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not word_scores:
            # Find any unused word
            unused_words = [w for w in self.word_list if w not in self.past_guesses]
            if unused_words:
                return random.choice(unused_words)
            else:
                # Truly stuck - just pick random
                return random.choice(self.word_list)
        
        # After many attempts, start exploring more aggressively
        if self.current_attempt > 7:
            exploration_chance = 0.2
            if random.random() < exploration_chance and len(word_scores) >= 5:
                # Choose from top candidates with higher probability for higher ranked words
                top_candidates = word_scores[:5]
                weights = [1.5, 1.2, 1.0, 0.8, 0.5][:len(top_candidates)]
                words = [w for w, _ in top_candidates]
                return random.choices(words, weights=weights, k=1)[0]
        
        return word_scores[0][0]

    def reset(self):
        """Reset the agent for a new game, keeping some long-term learning."""
        self.possible_words = self.word_list.copy()
        self.not_in_word = set()
        self.gray_letters = set()
        self.green_letters = {}
        self.yellow_letters = defaultdict(set)
        self.known_present_letters = set()
        
        # Reset guess tracking
        self.past_guesses = []
        self.rejected_words = set()
        self.repeat_guess_count = 0
        self.current_attempt = 0
        
        # Reset feedback consistency tracking
        self.feedback_consistency = {}
        self.letter_feedback_history = defaultdict(list)
        
        # Use a gentler decay rate for letter inconsistency
        for letter in self.letter_inconsistency:
            self.letter_inconsistency[letter] = max(0, self.letter_inconsistency[letter] - 1)
        
        # Use a gentler decay rate for priors
        decay_rate = 0.5  # Much less aggressive forgetting (was 0.1)
        min_prior = 1.0
        
        for key in self.letter_position_prior:
            alpha, beta = self.letter_position_prior[key]
            new_alpha = max(min_prior, alpha * decay_rate) if alpha > min_prior else alpha
            new_beta = max(min_prior, beta * decay_rate) if beta > min_prior else beta
            self.letter_position_prior[key] = (new_alpha, new_beta)
            
        for key in self.letter_in_word_prior:
            alpha, beta = self.letter_in_word_prior[key]
            new_alpha = max(min_prior, alpha * decay_rate) if alpha > min_prior else alpha
            new_beta = max(min_prior, beta * decay_rate) if beta > min_prior else beta
            self.letter_in_word_prior[key] = (new_alpha, new_beta)
            
        # Recover trust more aggressively between games
        self.trust_yellow = min(0.9, self.trust_yellow * 1.1)  # More aggressive recovery
        self.trust_gray = min(0.9, self.trust_gray * 1.1)      # More aggressive recovery
        
        # Gradually recover letter-specific confidence
        for letter in self.gray_confidence:
            self.gray_confidence[letter] = min(0.9, self.gray_confidence[letter] * 1.1)
        
        # Reset confidence threshold
        self.confidence_threshold = 0.4

    def adapt_to_deception(self, reward):
        """Adapt trust parameters based on game success"""
        if reward > 0:  # Successful game
            # If we succeeded, our trust model might be working
            # More aggressively increase trust in our model
            self.trust_yellow = min(0.9, self.trust_yellow * 1.05)  # More aggressive recovery
            self.trust_gray = min(0.9, self.trust_gray * 1.05)
            # Reduce exploration slightly
            self.exploration_weight = max(0.3, self.exploration_weight * 0.95)
            
            # Recover letter-specific confidence more for successful games
            for letter in self.gray_confidence:
                self.gray_confidence[letter] = min(0.9, self.gray_confidence[letter] * 1.05)
                
        else:  # Failed game
            # If we failed, maybe we trusted too much - but make reduction gentler
            self.trust_yellow = max(self.trust_floor, self.trust_yellow * 0.98)  # Gentler reduction
            self.trust_gray = max(self.trust_floor, self.trust_gray * 0.98)
            # Increase exploration but with a cap
            self.exploration_weight = min(self.exploration_cap, self.exploration_weight * 1.02)
            
            # Reduce confidence in gray letters more aggressively for failed games
            for letter in self.gray_confidence:
                self.gray_confidence[letter] = max(self.trust_floor, self.gray_confidence[letter] * 0.95)

def run_experiments_multiple_deception(word_list, deception_levels=[0.0, 0.05, 0.1, 0.15, 0.2], 
                                       num_episodes=100, max_attempts=12):
    """Run experiments across multiple deception levels and gather data for comparison plots"""
    results = {}
    
    for deception_prob in deception_levels:
        print(f"\n=== Running experiment with {deception_prob*100}% deception probability ===")
        
        env = WordleEnv(word_list, max_attempts=max_attempts, deception_prob=deception_prob)
        agent = BayesianWordleAgent(word_list)
        
        win_count = 0
        total_reward = 0
        total_attempts = 0
        
        for episode in tqdm(range(num_episodes), desc=f"Testing d={deception_prob}"):
            env.reset()
            agent.reset()
            
            done = False
            ep_reward = 0
            attempts = 0
            
            while not done:
                action = agent.select_action()
                next_history, reward, done, _ = env.step(action)
                guess, feedback = next_history[-1]
                agent.update_beliefs(guess, feedback)
                ep_reward += reward
                attempts += 1
                
            # Record results
            if ep_reward > 0:
                win_count += 1
            total_reward += ep_reward
            total_attempts += attempts
            
            # Adapt agent based on game outcome
            agent.adapt_to_deception(ep_reward)
        
        # Calculate summary statistics
        win_rate = win_count / num_episodes
        avg_reward = total_reward / num_episodes
        avg_attempts = total_attempts / num_episodes
        
        results[deception_prob] = {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_attempts': avg_attempts
        }
        
        print(f"Deception: {deception_prob}, Win Rate: {win_rate:.4f}, Avg Reward: {avg_reward:.4f}, Avg Attempts: {avg_attempts:.4f}")
    
    return results

def plot_deception_comparison(results, deception_levels):
    """Generate comparison plots for different deception levels"""
    if not results:
        print("No results to plot.")
        return
    
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Win Rate
    plt.subplot(1, 3, 1)
    win_rates = [results[d]['win_rate'] for d in deception_levels if d in results]
    plt.plot(deception_levels, win_rates, marker='o', linestyle='-', color='blue')
    plt.xlabel('Deception Probability')
    plt.ylabel('Win Rate')
    plt.title('Win Rate vs. Deception')
    plt.grid(True)
    
    # Subplot 2: Average Attempts
    plt.subplot(1, 3, 2)
    avg_attempts = [results[d]['avg_attempts'] for d in deception_levels if d in results]
    plt.plot(deception_levels, avg_attempts, marker='o', linestyle='-', color='red')
    plt.xlabel('Deception Probability')
    plt.ylabel('Average Attempts')
    plt.title('Attempts vs. Deception')
    plt.grid(True)
    
    # Subplot 3: Average Reward
    plt.subplot(1, 3, 3)
    avg_rewards = [results[d]['avg_reward'] for d in deception_levels if d in results]
    plt.plot(deception_levels, avg_rewards, marker='o', linestyle='-', color='green')
    plt.xlabel('Deception Probability')
    plt.ylabel('Average Reward')
    plt.title('Reward vs. Deception')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('deception_vs_performance.png')
    plt.close()
    print("Comparison plot saved to: deception_vs_performance.png")

def test_bayesian_agent(word_list, num_episodes=1000, deception_prob=0.1, max_attempts=12, smoothing_window=50):
    """Run a test on the Bayesian agent with specified deception probability."""
    
    print(f"\n=== Testing Bayesian agent with {deception_prob*100}% deception probability ===")
    
    # Initialize environment and agent
    env = WordleEnv(word_list, max_attempts=max_attempts, deception_prob=deception_prob)
    agent = BayesianWordleAgent(word_list)
    
    # Data collectors
    rewards_per_episode = []
    attempts_per_episode = []
    first_guesses = []
    trust_yellow_values = []
    trust_gray_values = []
    exploration_values = []
    confidence_thresholds = []
    possible_word_counts = []
    
    # Running window metrics for real-time tracking
    window_size = 50  
    running_win_rates = []
    episode_numbers = []
    
    # Start test
    start_time = time.time()
    
    win_count = 0  # Track for running win rate
    for episode in tqdm(range(num_episodes), desc=f"Testing (d={deception_prob}, att={max_attempts})"):
        # Reset environment
        episode_history = env.reset()
        agent.reset()  
        
        # Run a complete game
        done = False
        total_reward = 0
        attempt = 0
        first_guess_made = False
        
        while not done:
            # Store possible word count at each step
            possible_word_counts.append(len(agent.possible_words))
            
            action = agent.select_action()
            if not first_guess_made:
                first_guesses.append(action)
                first_guess_made = True
                
            next_history, reward, done, _ = env.step(action)
            guess, feedback = next_history[-1]
            agent.update_beliefs(guess, feedback)
            total_reward += reward
            attempt += 1
        
        # Record metrics
        rewards_per_episode.append(total_reward)
        attempts_per_episode.append(attempt)
        trust_yellow_values.append(agent.trust_yellow)
        trust_gray_values.append(agent.trust_gray)
        exploration_values.append(agent.exploration_weight)
        confidence_thresholds.append(agent.confidence_threshold)
        
        # Update win count and running win rate
        if total_reward > 0:
            win_count += 1
        
        # Calculate running win rate every window_size episodes
        if (episode + 1) % window_size == 0 or episode == num_episodes - 1:
            running_win_rate = win_count / (episode + 1)
            running_win_rates.append(running_win_rate)
            episode_numbers.append(episode + 1)
            
            # Print progress update
            print(f"Episodes {episode+1}/{num_episodes}, Running Win Rate: {running_win_rate:.4f}, "
                  f"Trust Yellow: {agent.trust_yellow:.4f}, Trust Gray: {agent.trust_gray:.4f}")
        
        # Adapt trust parameters based on game outcome
        agent.adapt_to_deception(total_reward)
    
    # Calculate final statistics
    final_win_rate = win_count / num_episodes
    avg_reward = sum(rewards_per_episode) / num_episodes
    avg_attempts = sum(attempts_per_episode) / num_episodes
    
    winning_attempts = [a for a, r in zip(attempts_per_episode, rewards_per_episode) if r > 0]
    losing_attempts = [a for a, r in zip(attempts_per_episode, rewards_per_episode) if r <= 0]
    
    attempt_distribution_wins = Counter(winning_attempts)
    attempt_distribution_losses = Counter(losing_attempts)
    
    # Print summary statistics
    print("\n=== Test Results ===")
    print(f"Deception Probability: {deception_prob}")
    print(f"Episodes: {num_episodes}")
    print(f"Win Rate: {final_win_rate:.4f}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average Attempts: {avg_attempts:.4f}")
    print(f"Final Trust Yellow: {agent.trust_yellow:.4f}")
    print(f"Final Trust Gray: {agent.trust_gray:.4f}")
    print(f"Final Exploration Weight: {agent.exploration_weight:.4f}")
    print(f"Average Possible Words Count: {sum(possible_word_counts)/len(possible_word_counts) if possible_word_counts else 0:.2f}")
    
    if winning_attempts:
        avg_win_attempts = sum(winning_attempts) / len(winning_attempts)
        print(f"Average Attempts (wins only): {avg_win_attempts:.4f}")
    
    if losing_attempts:
        avg_loss_attempts = sum(losing_attempts) / len(losing_attempts)
        print(f"Average Attempts (losses only): {avg_loss_attempts:.4f}")
    
    # Calculate test duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Test completed in {duration:.2f} seconds")
    
    # Generate plots
    
    # 1. Learning Curve - Win Rate over Episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episode_numbers, running_win_rates, marker='o', linestyle='-', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Running Win Rate')
    plt.title(f'Learning Progress - Win Rate Over Episodes ({deception_prob*100}% Deception)')
    plt.grid(True)
    plt.savefig(f'learning_curve_win_rate_d{deception_prob}.png')
    plt.close()
    print(f"Learning curve plot saved to: learning_curve_win_rate_d{deception_prob}.png")
    
    # 2. Rewards per Episode with Moving Average
    plt.figure(figsize=(12, 6))
    episodes = range(1, num_episodes + 1)
    rewards_series = pd.Series(rewards_per_episode)
    moving_avg = rewards_series.rolling(window=smoothing_window, min_periods=1).mean()
    plt.plot(episodes, rewards_per_episode, alpha=0.3, label='Reward per Episode', color='blue')
    plt.plot(episodes, moving_avg, label=f'Moving Avg (window={smoothing_window})', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Rewards per Episode with Moving Average ({deception_prob*100}% Deception)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'rewards_with_moving_avg_d{deception_prob}.png')
    plt.close()
    print(f"Rewards plot saved to: rewards_with_moving_avg_d{deception_prob}.png")
    
    # Save the results data
    results_data = {
        'deception_prob': deception_prob,
        'num_episodes': num_episodes,
        'win_rate': final_win_rate,
        'avg_reward': avg_reward,
        'avg_attempts': avg_attempts,
        'rewards_per_episode': rewards_per_episode,
        'attempts_per_episode': attempts_per_episode,
        'running_win_rates': running_win_rates,
        'episode_numbers': episode_numbers
    }
    
    with open(f'test_results_d{deception_prob}_e{num_episodes}.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    return results_data

def visualize_example_games(word_list, num_games=3, deception_prob=0.1, max_attempts=12):
    """Run and visualize a few example games to see the agent in action."""
    GREEN = '\033[92m'; YELLOW = '\033[93m'; GRAY = '\033[90m'; RESET = '\033[0m'
    
    print(f"\n=== Visualizing {num_games} Example Games with {deception_prob*100}% Deception ===")
    
    env = WordleEnv(word_list, max_attempts=max_attempts, deception_prob=deception_prob)
    agent = BayesianWordleAgent(word_list)
    
    games_data = []
    all_possible_word_counts = []
    
    for game in range(num_games):
        print(f"\n----- Game {game+1} -----")
        history = env.reset()
        agent.reset()
        
        # Tracking data for this game
        game_data = {
            'game_number': game + 1,
            'solution': env.solution,
            'attempts': [],
            'possible_words_count': [len(agent.possible_words)],
        }
        
        done = False
        attempt = 0
        
        while not done:
            action = agent.select_action()
            next_history, reward, done, _ = env.step(action)
            guess, feedback = next_history[-1]
            
            # Format the guess with colors
            colored_guess = "".join(
                f"{GREEN}{L}{RESET}" if fb == "green" else
                f"{YELLOW}{L}{RESET}" if fb == "yellow" else
                f"{GRAY}{L}{RESET}"
                for L, fb in zip(guess.upper(), feedback)
            )
            
            # Record before updating beliefs
            attempt += 1
            
            # Store attempt data
            game_data['attempts'].append({
                'number': attempt,
                'guess': guess,
                'feedback': feedback,
                'colored_guess': colored_guess
            })
            
            # Update agent beliefs
            agent.update_beliefs(guess, feedback)
            
            # Record possible word count after update
            possible_words_count = len(agent.possible_words)
            game_data['possible_words_count'].append(possible_words_count)
            all_possible_word_counts.append(possible_words_count)
            
            # Print information about this attempt
            print(f"Attempt {attempt}: {colored_guess} -> Possible words: {possible_words_count}")
            
            # Show past guesses
            print(f"  Past guesses: {agent.past_guesses}")
            
            # Show possible words if few remain
            if possible_words_count <= 10: 
                print(f"  Candidates: {agent.possible_words}")
        
        # Game result
        game_data['result'] = 'WON' if reward > 0 else 'LOST'
        game_data['attempts_count'] = attempt
        
        if reward > 0:
            print(f"Result: Agent WON in {attempt} attempts!")
        else:
            print(f"Result: Agent LOST after {attempt} attempts. Solution was: {env.solution}")
        
        games_data.append(game_data)
        
        # Plot possible word count reduction for this game
        plt.figure(figsize=(10, 6))
        attempt_numbers = range(attempt + 1)  # 0 to final attempt
        plt.plot(attempt_numbers, game_data['possible_words_count'], marker='o', linestyle='-', color='purple')
        plt.yscale('log')  # Log scale
        plt.xlabel('Attempt Number')
        plt.ylabel('Number of Possible Words (log scale)')
        plt.title(f'Candidate Word Reduction - Game {game+1} ({deception_prob*100}% Deception)')
        plt.grid(True)
        plt.savefig(f'word_reduction_game{game+1}_d{deception_prob}.png')
        plt.close()
        print(f"Word reduction plot saved to: word_reduction_game{game+1}_d{deception_prob}.png")
    
    print("\n--- Example Games Completed ---")
    
    # Save games data
    with open(f'example_games_d{deception_prob}.pkl', 'wb') as f:
        pickle.dump(games_data, f)

if __name__ == "__main__":
    # Configuration
    WORD_LIST_FILE = "wordList_2315.txt"  # Adjust to your file name
    NUM_EPISODES = 10000  # For main test
    COMPARISON_EPISODES = 100  # For multi-deception comparison (faster)
    DECEPTION_PROB = 0.1  # Main test deception level
    DECEPTION_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2]  # For comparison plots
    MAX_ATTEMPTS = 12
    SMOOTHING_WINDOW = 50
    NUM_EXAMPLE_GAMES = 3
    
    # Load word list
    word_list = load_word_list(WORD_LIST_FILE)
    if not word_list:
        print("Failed to load word list. Exiting.")
        exit()
    
    print(f"Loaded {len(word_list)} words from {WORD_LIST_FILE}.")
    
    # Choose which tests to run
    run_main_test = True  # Main 1000-episode test with 10% deception
    run_comparison = True  # Multi-deception level comparison (100 episodes each)
    run_examples = True   # Example game visualizations
    
    # Run the main test (1000 episodes with 10% deception)
    if run_main_test:
        print("\n--- Starting Main Bayesian Agent Test (1000 episodes) ---")
        main_results = test_bayesian_agent(
            word_list,
            num_episodes=NUM_EPISODES,
            deception_prob=DECEPTION_PROB,
            max_attempts=MAX_ATTEMPTS,
            smoothing_window=SMOOTHING_WINDOW
        )
    
    # Run comparison across deception levels
    if run_comparison:
        print("\n--- Starting Deception Level Comparison Tests ---")
        comparison_results = run_experiments_multiple_deception(
            word_list,
            deception_levels=DECEPTION_LEVELS,
            num_episodes=COMPARISON_EPISODES,
            max_attempts=MAX_ATTEMPTS
        )
        plot_deception_comparison(comparison_results, DECEPTION_LEVELS)
    
    # Visualize example games
    if run_examples:
        visualize_example_games(
            word_list, 
            num_games=NUM_EXAMPLE_GAMES,
            deception_prob=DECEPTION_PROB,
            max_attempts=MAX_ATTEMPTS
        )
    
    print("\n--- All Tests Completed ---")