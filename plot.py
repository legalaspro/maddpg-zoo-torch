# plot_rewards.py
import argparse
import os
import numpy as np

import utils.plotting as plotting
from utils.env import get_env_info

def parse_args():
    parser = argparse.ArgumentParser(description='Plot and compare algorithm results')
    parser.add_argument('--mode', type=str, choices=['single', 'compare'], default='single',
                        help='Plot mode: single algorithm or comparison')
    parser.add_argument('--env-name', type=str, required=True, 
                        help='Environment name')
    
    # Arguments for comparison plotting
    parser.add_argument('--algo-name', type=str, default='MADDPG',
                        help='Name of algorithm (for compare mode)')
    parser.add_argument('--rewards-path', type=str,
                        help='Path to first algorithm results (for compare mode)')
    parser.add_argument('--algo2-name', type=str, default='MADDPG-Approx',
                        help='Name of second algorithm (for compare mode)')
    parser.add_argument('--rewards2-path', type=str,
                        help='Path to second algorithm results (for compare mode)')
    
    # Shared arguments
    parser.add_argument('--window-size', type=int, default=100,
                        help='Window size for running average')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Directory to save plots')
    parser.add_argument('--target-score', type=int, default=None,
                        help='Target score for single mode')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Get environment information
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name,
        apply_padding=False 
    )


    if args.mode == 'single':
        output_dir = f"{args.output_dir}/{args.env_name}/{args.algo_name}"
        os.makedirs(output_dir, exist_ok=True)
        agent_rewards = np.load(args.rewards_path, allow_pickle=True)
        
        plotting.plot_rewards_single_env(agents, agent_rewards, output_dir, 
            env_name=args.env_name, target_score=args.target_score, window_size=args.window_size)
    elif args.mode == 'compare':
        output_dir = f"{args.output_dir}/compare/{args.env_name}"
        os.makedirs(output_dir, exist_ok=True)
        algo_paths = {
            args.algo_name: args.rewards_path,
            args.algo2_name: args.rewards2_path
        }
        plotting.compare_algorithms(args.env_name, algo_paths, output_dir, window_size=args.window_size)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    


    

   

    

