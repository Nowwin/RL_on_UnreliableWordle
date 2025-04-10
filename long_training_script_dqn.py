#!/usr/bin/env python
"""
This script runs a long training session for the Wordle DQN agent.
It includes checkpointing and the ability to resume training.
"""

import argparse
import torch
import numpy as np
import random
import os
import time
from dqn import (
    WordleAgent, WordleEnv, encode_history, 
    train_dqn_agent, evaluate_agent, plot_detailed_results,Experience,namedtuple
)

def load_word_list(filename):
    """Load a list of 5-letter words from a text file."""
    with open(filename, 'r') as f:
        return [line.strip().lower() for line in f if len(line.strip()) == 5]

def train_with_checkpoints(word_list, args):
    """
    Train a DQN agent with regular checkpoints and the ability to resume training.
    
    Args:
        word_list: List of words for the Wordle environment
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate state dimensions
    state_dim = (5 * 3) + 2 + (26 * 5 * 2) + 26 + 26
    action_dim = len(word_list)
    
    all_results = {}
    
    for deception_prob in args.deception_levels:
        print(f"\n=== Running experiment with deception probability {deception_prob} ===")
        
        # Create environment and agent
        env = WordleEnv(word_list, max_attempts=6, deception_prob=deception_prob)
        agent = WordleAgent(state_dim, action_dim, word_list, 
                           hidden_dim=args.hidden_dim, lr=args.learning_rate)
        
        # File paths for saving/loading
        checkpoint_dir = os.path.join(args.output_dir, f"deception_{deception_prob}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        
        # Resume from checkpoint if available and requested
        start_episode = 0
        if args.resume and os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            agent.q_network.load_state_dict(checkpoint['q_network'])
            agent.target_network.load_state_dict(checkpoint['target_network'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                agent.scheduler.load_state_dict(checkpoint['scheduler'])
            
            start_episode = checkpoint['episode'] + 1
            print(f"Resuming from episode {start_episode}")
        
        # Training parameters
        batch_size = args.batch_size
        gamma = args.gamma
        epsilon_start = args.epsilon_start
        epsilon_end = args.epsilon_end
        epsilon_decay = args.epsilon_decay
        target_update = args.target_update
        tau = args.tau  # Soft update parameter
        
        # For tracking training progress
        rewards = []
        win_rates = []
        losses = []
        epsilon = epsilon_start
        win_count = 0
        episodes_without_improvement = 0
        best_win_rate = 0
        
        # For periodic evaluation
        eval_results = []
        
        # Training loop
        print(f"Starting training for {args.num_episodes} episodes (starting from {start_episode})")
        training_start_time = time.time()
        
        for episode in range(start_episode, args.num_episodes):
            # Reset environment
            history = env.reset()
            state = encode_history(history, word_list, agent.letter_freq)
            total_reward = 0
            done = False
            
            # Play one episode
            while not done:
                # Select action
                action_idx, action_word = agent.get_action(state, epsilon, history)
                
                # Take step in environment
                next_history, reward, done, _ = env.step(action_word)
                next_state = encode_history(next_history, word_list, agent.letter_freq)
                
                # Store experience
                agent.replay_buffer.push(Experience(state, action_idx, reward, next_state, done))
                
                # Update network multiple times
                for _ in range(4):
                    loss = agent.update(batch_size, gamma)
                    if loss > 0:
                        losses.append(loss)
                
                # Move to next state
                state = next_state
                history = next_history
                total_reward += reward
            
            # Soft update target network
            agent.update_target_network(tau)
            
            # Hard update target network periodically
            if episode % target_update == 0:
                agent.update_target_network(1.0)
            
            # Update epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Record stats
            rewards.append(total_reward)
            win = (total_reward > 0)
            win_count += int(win)
            current_win_rate = win_count / (episode - start_episode + 1)
            win_rates.append(current_win_rate)
            
            # Learning rate scheduling
            if episode % 100 == 0:
                agent.scheduler.step(current_win_rate)
            
            # Print progress
            if episode % args.print_every == 0:
                elapsed_time = time.time() - training_start_time
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                
                print(f"Episode {episode}/{args.num_episodes} | "
                      f"Time: {hours}h {minutes}m | "
                      f"Reward: {total_reward:.2f} | "
                      f"Win Rate: {current_win_rate:.4f} | "
                      f"Epsilon: {epsilon:.4f}")
                
                # Recent stats
                recent_episodes = min(100, episode - start_episode + 1)
                if recent_episodes > 0:
                    recent_rewards = rewards[-recent_episodes:]
                    recent_win_rate = sum(1 for r in recent_rewards if r > 0) / recent_episodes
                    print(f"Recent {recent_episodes} episodes win rate: {recent_win_rate:.4f}")
            
            # Save checkpoint
            if episode % args.checkpoint_every == 0 and episode > 0:
                checkpoint = {
                    'episode': episode,
                    'q_network': agent.q_network.state_dict(),
                    'target_network': agent.target_network.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'scheduler': agent.scheduler.state_dict(),
                    'win_rate': current_win_rate,
                    'epsilon': epsilon
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at episode {episode}")
            
            # Periodic evaluation
            if episode % args.eval_every == 0 and episode > 0:
                print("\nRunning evaluation...")
                eval_stats = evaluate_agent(
                    agent, env, num_episodes=args.eval_episodes, render_every=0
                )
                eval_results.append({
                    'episode': episode,
                    'stats': eval_stats
                })
                
                print(f"Evaluation at episode {episode}:")
                print(f"  Win Rate: {eval_stats['win_rate']:.4f}")
                print(f"  Avg Reward: {eval_stats['avg_reward']:.4f}")
                print(f"  Avg Attempts: {eval_stats['avg_attempts']:.4f}")
                
                # Save if best
                if eval_stats['win_rate'] > best_win_rate:
                    best_win_rate = eval_stats['win_rate']
                    episodes_without_improvement = 0
                    # Save best model
                    agent.save(best_model_path)
                    print(f"New best model saved with win rate: {best_win_rate:.4f}")
                else:
                    episodes_without_improvement += args.eval_every
            
            # Early stopping check
            if episodes_without_improvement >= args.early_stopping:
                print(f"Early stopping: No improvement for {episodes_without_improvement} episodes")
                break
            
            # Win rate threshold check
            if current_win_rate >= args.win_threshold:
                print(f"Win rate threshold {args.win_threshold} reached!")
                break
        
        # Final evaluation
        print("\nRunning final evaluation...")
        final_eval = evaluate_agent(agent, env, num_episodes=args.eval_episodes * 2)
        
        # Save results
        all_results[deception_prob] = {
            'train': {
                'rewards': rewards,
                'win_rates': win_rates,
                'losses': losses,
                'final_win_rate': win_rates[-1] if win_rates else 0,
                'episodes_trained': episode - start_episode + 1
            },
            'eval': final_eval
        }
        
        # Print final results
        print(f"\nFinal results for deception probability {deception_prob}:")
        print(f"  Win Rate: {final_eval['win_rate']:.4f}")
        print(f"  Avg Reward: {final_eval['avg_reward']:.4f}")
        print(f"  Avg Attempts: {final_eval['avg_attempts']:.4f}")
        
        if final_eval['attempt_distribution']:
            print("  Attempt distribution for winning games:")
            for attempt, count in sorted(final_eval['attempt_distribution'].items()):
                percentage = count/sum(final_eval['attempt_distribution'].values())*100
                print(f"    {attempt} attempts: {count} games ({percentage:.1f}%)")
    
    # Plot overall results
    plot_detailed_results(all_results, args.deception_levels)
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Wordle DQN agent for many episodes')
    
    # Data parameters
    parser.add_argument('--word_list', type=str, default='wordList.txt', 
                      help='Path to word list file')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000, 
                      help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=128, 
                      help='Batch size for DQN updates')
    parser.add_argument('--hidden_dim', type=int, default=256, 
                      help='Hidden dimension for neural network')
    parser.add_argument('--learning_rate', type=float, default=0.0005, 
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, 
                      help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, 
                      help='Starting epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01, 
                      help='Final epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.99995, 
                      help='Decay rate for epsilon')
    parser.add_argument('--target_update', type=int, default=10, 
                      help='Episodes between target network updates')
    parser.add_argument('--tau', type=float, default=0.01, 
                      help='Soft update parameter')
    
    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=500, 
                      help='Number of episodes for evaluation')
    parser.add_argument('--eval_every', type=int, default=5000, 
                      help='Episodes between evaluations')
    parser.add_argument('--win_threshold', type=float, default=0.5, 
                      help='Win rate threshold for early stopping')
    parser.add_argument('--early_stopping', type=int, default=20000, 
                      help='Stop if no improvement for this many episodes')
    
    # Experiment parameters
    parser.add_argument('--deception_levels', type=float, nargs='+', default=[0.0, 0.05, 0.1], 
                      help='Deception probabilities to test')
    
    # Utility parameters
    parser.add_argument('--output_dir', type=str, default='results', 
                      help='Directory to save results')
    parser.add_argument('--checkpoint_every', type=int, default=1000, 
                      help='Episodes between checkpoints')
    parser.add_argument('--print_every', type=int, default=100, 
                      help='Episodes between printing progress')
    parser.add_argument('--resume', action='store_true', 
                      help='Resume training from checkpoint if available')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load word list
    print(f"Loading word list from {args.word_list}...")
    word_list = load_word_list(args.word_list)
    print(f"Loaded {len(word_list)} words")
    
    # Start training
    results = train_with_checkpoints(word_list, args)