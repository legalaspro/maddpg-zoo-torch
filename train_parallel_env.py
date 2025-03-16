"""
Training script for MADDPG with PettingZoo using parallel environments
"""
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import time

from maddpg import MADDPG, ReplayBuffer
from utils.env import get_env_info, ENV_MAP, create_single_env, create_parallel_env
from utils.logger import Logger
from utils.utils import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="simple_adversary_v3", 
                       choices=list(ENV_MAP.keys()),
                       help="Name of the environment to use")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--total-steps", type=int, default=int(1e6), help="Number of steps")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum steps per episode")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter")
    parser.add_argument("--actor-lr", type=float, default=5e-4, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=5e-4, help="Critic learning rate")
    parser.add_argument("--hidden-sizes", type=str, default="64,64", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--update-every", type=int, default=1, help="Update networks every n steps")
    parser.add_argument("--num-updates", type=int, default=4, help="Number of updates per step (important for parallel environments)")
    parser.add_argument("--noise-scale", type=float, default=0.3, help="Initial Gaussian noise scale")
    parser.add_argument("--min-noise", type=float, default=0.05, help="Minimum Gaussian noise scale")
    parser.add_argument("--noise-decay-steps", type=int, 
                        default=int(5e5),
                        help="Number of step to decay noise to min_noise default: 500k")
    parser.add_argument("--use-noise-decay", action="store_true", help="Use noise decay")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluate every n steps")
    parser.add_argument("--render-mode", type=str, default=None, choices=[None, "human", "rgb_array"], 
                       help="Render mode for visualization")
    parser.add_argument("--create-gif", action="store_true", help="Create GIF of episodes")
    return parser.parse_args()

def train(args):
  
    # Add timestamp to experiment name for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (
        "parallel"
        f"{args.batch_size}"
        f"_steps{args.max_steps}"
        f"_envs{args.num_envs}"
        f"_g{args.gamma}"
        f"_t{args.tau}"
        f"_alr{args.actor_lr}"
        f"_clr{args.critic_lr}"
        f"_noise{args.noise_scale}"
        f"_{timestamp}")

    logger = Logger(
        run_name=experiment_name,
        folder="runs",
        algo="MADDPG",
        env=args.env_name
    )
    logger.log_all_hyperparameters(vars(args))
    
    # Get environment information
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name, 
        max_steps=args.max_steps,
        apply_padding=True  # We need padding for parallel environments
    )
    
    # Create parallel environments
    num_envs = max(1, args.num_envs)  # Ensure at least 1 environment
    env = create_parallel_env(
        env_name=args.env_name,
        max_steps=args.max_steps, 
        num_envs=num_envs
    )

    # Create evaluation environment
    env_evaluate = create_single_env(
        env_name=args.env_name,
        max_steps=args.max_steps,
        render_mode="rgb_array",
        apply_padding=True
    )
    
    # Model path
    model_path = os.path.join(logger.dir_name, "model.pt")
    best_model_path = os.path.join(logger.dir_name, "best_model.pt")
    best_score = -float('inf')
    
    # Parse hidden sizes
    hidden_sizes = tuple(map(int, args.hidden_sizes.split(',')))
    
    # Create MADDPG agent with padded state sizes
    maddpg = MADDPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        hidden_sizes=hidden_sizes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        action_low=action_low,
        action_high=action_high
    )
    
    # Create replay buffer with the correct dimensions
    buffer = ReplayBuffer(
        buffer_size=min(args.buffer_size, args.total_steps),
        batch_size=args.batch_size,
        agents=agents,
        state_sizes=state_sizes,
        action_sizes=action_sizes
    )
    
    print(f"Starting training with experiment name: {experiment_name}")
    
    # Training loop
    noise_scale = args.noise_scale
    decay_steps = min(args.noise_decay_steps // num_envs, args.total_steps // num_envs)
    noise_decay = (args.noise_scale - args.min_noise) / decay_steps
    print(f"Using linear noise decay: {args.noise_scale} to {args.min_noise} over {decay_steps} steps")
    print(f"Noise will decrease by {noise_decay:.6f} per step")
    
    # For tracking agent-specific rewards
    agent_rewards = [[] for _ in range(len(agents))]
    global_step = 0

    eval_interval = max(1, (args.eval_interval // num_envs) * num_envs)

    evaluate(env_evaluate, maddpg, logger, record_gif=args.create_gif, num_eval_episodes=10, global_step=0)
    
    # Create a custom tqdm that shows the actual episode count
    with tqdm(total=args.total_steps, desc="Training") as pbar:
        while global_step < args.total_steps:
            
            # Reset environment
            observations, _ = env.reset()
            # Reshape observations to (num_envs, num_agents, obs_dim)
            observations_reshaped = observations.reshape(num_envs, num_agents, -1)  # -1 infers obs_dim

            episode_rewards = np.zeros((num_envs, num_agents))  # Shape: (num_envs, num_agents)

            if global_step == 0:
                print(f"Observations shape: {observations.shape}")
                print(f"Reshaped Observations shape: {observations_reshaped.shape}")
            
             # Run one episode (up to max_steps or done)
            for step in range(args.max_steps):

                # Batch states for all agents across environments
                states_batched = [observations_reshaped[:, i, :] for i in range(num_agents)]  # List of (4, obs_dim)
                
                # Get batched actions (requires maddpg.act to handle batched inputs)
                actions_batched = maddpg.act(states_batched, add_noise=True, noise_scale=noise_scale)  # List of (4, action_dim)
                
                # Stack and reshape actions to (num_envs * num_agents, action_dim)
                actions_stacked = np.stack(actions_batched, axis=1)  # (4, 3, action_dim) = (num_envs, num_agents, action_dim)
                actions_array = actions_stacked.reshape(num_envs * num_agents, -1)  # (12, action_dim) = (num_envs * num_agents, action_dim)    
                
                # Debug logging
                if global_step == 0:
                    print(f"Actions array shape: {actions_array.shape}")
                    print(f"States batched shapes: {[s.shape for s in states_batched]}")
                    print(f"Actions batched shapes: {[a.shape for a in actions_batched]}")

                # Step all environments
                next_observations, rewards, terminations, truncations, infos = env.step(actions_array)
                dones = np.logical_or(terminations, truncations)
        
                # Reshape outputs
                next_observations_reshaped = next_observations.reshape(num_envs, num_agents, -1)  # (4, 3, obs_dim) = (num_envs, num_agents, obs_dim)   
                rewards_reshaped = rewards.reshape(num_envs, num_agents)  # (4, 3) = (num_envs, num_agents)
                # we care about the termination of the episode
                terminations_reshaped = terminations.reshape(num_envs, num_agents)  # (4, 3) = (num_envs, num_agents)
                
                # Debug logging
                if global_step == 0:
                    print(f"Next observations shape: {next_observations.shape}")
                    print(f"Next observations reshaped shape: {next_observations_reshaped.shape}")
                    print(f"Rewards shape: {rewards.shape}")
                    print(f"Rewards reshaped shape: {rewards_reshaped.shape}")
                    print(f"Dones shape: {dones.shape}")
                    print(f"Dones reshaped shape: {terminations_reshaped.shape}")

                # Add experiences to buffer
                for env_idx in range(num_envs):
                    states = observations_reshaped[env_idx]  # Shape: (num_agents, obs_dim)
                    actions = actions_stacked[env_idx]  # Shape: (num_agents, action_dim)
                    rewards_env = rewards_reshaped[env_idx]  # Shape: (num_agents,)
                    next_states = next_observations_reshaped[env_idx]  # Shape: (num_agents, obs_dim)
                    dones_env = terminations_reshaped[env_idx]  # Shape: (num_agents,)
                    
                    # Debug logging for first environment
                    if global_step == 0 and env_idx == 0:
                        print(f"Env {env_idx} - States shape: {states.shape}")
                        print(f"Env {env_idx} - Actions shape: {actions.shape}")
                        print(f"Env {env_idx} - Rewards shape: {rewards_env.shape}")
                        print(f"Env {env_idx} - Next states shape: {next_states.shape}")
                        print(f"Env {env_idx} - Dones shape: {dones_env.shape}")
                    
                    buffer.add(
                        states=states,
                        actions=actions,
                        rewards=rewards_env,
                        next_states=next_states,
                        dones=dones_env
                    )

                # Update episode rewards vectorized
                episode_rewards += rewards_reshaped
            
                # Update observations
                observations_reshaped = next_observations_reshaped
                
                # Learn if enough samples are available
                if len(buffer) > args.batch_size and global_step % args.update_every == 0:
                    for _ in range(args.num_updates):
                        for i in range(num_agents):
                            experiences = buffer.sample()
                            critic_loss, actor_loss = maddpg.learn(experiences, i)
                            logger.add_scalar(f'{agents[i]}/critic_loss', critic_loss, global_step)
                            logger.add_scalar(f'{agents[i]}/actor_loss', actor_loss, global_step)
                        maddpg.update_targets() 
                
                if args.use_noise_decay:
                    noise_scale = max(
                        args.min_noise,
                        noise_scale - noise_decay
                    )
                
                # Increment global step for all envs
                global_step += num_envs
                pbar.update(num_envs)

                # Usually for the MPE they all finish at the same time
                if any(dones):
                    # print(f"One of the environments completed at step {step}")
                    break
            
            # Log rewards to TensorBoard
            for agent_idx in range(num_agents):
                # store the mean, min, max of the episode rewards for each agent
                agent_rewards[agent_idx].append([
                    np.mean(episode_rewards[:, agent_idx]), 
                    np.min(episode_rewards[:, agent_idx]), 
                    np.max(episode_rewards[:, agent_idx])])
                logger.add_scalar(f'{agents[agent_idx]}/episode_reward', np.mean(episode_rewards[:, agent_idx]), global_step)
            logger.add_scalar('noise/scale', noise_scale, global_step)
            logger.add_scalar('train/total_reward', np.sum(episode_rewards)/num_envs, global_step)
            
            if global_step % eval_interval == 0:
                maddpg.save(model_path)
                avg_eval_rewards = evaluate(env_evaluate, maddpg, logger,
                        num_eval_episodes=10, record_gif=args.create_gif, global_step=global_step)
                np.save(os.path.join(logger.dir_name, "agent_rewards.npy"), agent_rewards)
                score = np.sum(avg_eval_rewards)
                if score > best_score:
                    best_score = score
                    maddpg.save(best_model_path)
    
    # Save final models
    maddpg.save(model_path)
    
    # Close environment and TensorBoard writer
    env.close()
    env_evaluate.close()
    logger.close()

    np.save(os.path.join(logger.dir_name, "agent_rewards.npy"), agent_rewards)

    # Return both the agent rewards and the experiment name
    return agent_rewards, experiment_name

if __name__ == "__main__":
    args = parse_args()
    train(args) 
