#!/usr/bin/env python
"""
This script runs DQN experiments on the Wordle environment with different deception probabilities.
"""

import argparse
import torch
import numpy as np
import random
import os
from dqn import load_word_list, run_dqn_experiment, plot_results

def main():
    parser = argparse.ArgumentParser(description='Run DQN experiments on Wordle with different deception probabilities')
    parser.add_argument('--word_list', type=str, default='wordList.txt', help='Path to the word list file')
    parser.add_argument('--train_episodes', type=int, default=5000, help='Number of episodes to train for each deception level')
    parser.add_argument('--eval_episodes', type=int, default=1000, help='Number of episodes to evaluate for each deception level')
    parser.add_argument('--deception_levels', type=float, nargs='+', default=[0.0, 0.05, 0.1], 
                        help='Deception probabilities to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load the word list
    print(f"Loading word list from {args.word_list}...")
    word_list = load_word_list(args.word_list)
    print(f"Loaded {len(word_list)} words")
    
    # Run the experiments
    print(f"Running experiments with deception levels: {args.deception_levels}")
    results = run_dqn_experiment(
        word_list, 
        num_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        deception_levels=args.deception_levels
    )
    
    # Plot and save the results
    plot_results(results, args.deception_levels)
    print(f"Results saved to {args.output_dir}/dqn_results.png")
    
    # Print summary table
    print("\n=== Summary of Results ===")
    print("Deception | Win Rate | Avg Reward | Avg Attempts")
    print("-" * 50)
    for d in args.deception_levels:
        win_rate = results[d]['eval']['win_rate']
        avg_reward = results[d]['eval']['avg_reward']
        avg_attempts = results[d]['eval']['avg_attempts']
        print(f"{d:.2f}      | {win_rate:.4f}   | {avg_reward:.4f}    | {avg_attempts:.4f}")

if __name__ == "__main__":
    main()