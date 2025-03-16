"""
Training script for MADDPG with PettingZoo
"""
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime


from maddpg import MADDPG, MADDPGApprox, ReplayBuffer
from utils.env import get_env_info, ENV_MAP, create_single_env
from utils.logger import Logger
from utils.utils import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="simple_spread_v3", 
                       choices=list(ENV_MAP.keys()),
                       help="Name of the environment to use")
    parser.add_argument("--algo", type=str, default="MADDPG", choices=["MADDPG", "MADDPGApprox"],
                       help="Algorithm to use")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6), help="Total timesteps")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="Replay buffer size")
    parser.add_argument("--warmup-steps", type=int, default=20000, help="Warmup steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum steps per episode")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter")
    parser.add_argument("--actor-lr", type=float, default=1e-3, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=2e-3, help="Critic learning rate")
    parser.add_argument("--hidden-sizes", type=str, default="64,64", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--update-every", type=int, default=15, help="Update networks every n steps")
    parser.add_argument("--noise-scale", type=float, default=0.3, help="Initial noise scale")
    parser.add_argument("--min-noise", type=float, default=0.05, help="Minimum noise scale")
    parser.add_argument("--noise-decay-steps", type=int, 
                        default=int(3e5), 
                        help="Number of step to decay noise to min_noise default: 300k")
    parser.add_argument("--use-noise-decay", action="store_true", help="Use noise decay")
    parser.add_argument("--render-mode", type=str, default=None, choices=[None, "human", "rgb_array"], 
                       help="Render mode for visualization")
    parser.add_argument("--create-gif", action="store_true", help="Create GIF of episodes")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluate every n steps")
   
    return parser.parse_args()

def train(args):
    
    # Add timestamp to experiment name for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (
        "single"
        f"_b{args.batch_size}"
        f"_usteps{args.update_every}"
        f"_g{args.gamma}"
        f"_t{args.tau}"
        f"_alr{args.actor_lr}"
        f"_clr{args.critic_lr}"
        f"_n{args.noise_scale}"
        f"_minn{args.min_noise}"
        f"_h{args.hidden_sizes}"
        f"_{timestamp}")

    logger = Logger(
        run_name=experiment_name,
        folder="runs",
        algo=args.algo,
        env=args.env_name
    )
    logger.log_all_hyperparameters(vars(args))
    

    # Get environment information
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name, 
        max_steps=args.max_steps,
        apply_padding=False  
    )

    # Create environment with appropriate render mode
    env = create_single_env(
        env_name=args.env_name,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        apply_padding=False
    )
    
    # Create evaluation environment
    env_evaluate = create_single_env(
        env_name=args.env_name,
        max_steps=args.max_steps,
        render_mode="rgb_array",
        apply_padding=False
    )
    
    # Model path
    model_path = os.path.join(logger.dir_name, "model.pt")
    best_model_path = os.path.join(logger.dir_name, "best_model.pt")
    best_score = -float('inf')
    
    # Parse hidden sizes
    hidden_sizes = tuple(map(int, args.hidden_sizes.split(',')))
    
    # Create MADDPG agent
    if args.algo == "MADDPG":
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
    else:
        maddpg = MADDPGApprox(
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
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        agents=agents,
        state_sizes=state_sizes,
        action_sizes=action_sizes
    )
    
    # Training loop
    noise_scale = args.noise_scale
    noise_decay = (args.noise_scale - args.min_noise) / min(args.noise_decay_steps, args.total_timesteps)
    print(f"Using linear noise decay: {args.noise_scale} to {args.min_noise} over {args.noise_decay_steps} steps")
    print(f"Noise will decrease by {noise_decay:.6f} per step")

    evaluate(env_evaluate, maddpg, logger, record_gif=args.create_gif, num_eval_episodes=10, global_step=0)
    
    # For tracking agent-specific rewards
    agent_rewards = [[] for _ in range(len(agents))]
    episode_rewards = np.zeros(len(agents))

    # Reset environment and agents
    observations, _ = env.reset()

    for global_step in tqdm(range(1, args.total_timesteps + 1), desc="Training"):
        
        # Get states for all agents
        states_list = [np.array(observations[agent], dtype=np.float32) for agent in agents]
        
        # Get actions for all agents
        actions_list = maddpg.act(states_list, add_noise=True, noise_scale=noise_scale)
        
        # Convert actions to dictionary for environment
        actions = {agent: action for agent, action in zip(agents, actions_list)}
        
        # Take a step in the environment
        next_observations, rewards, terminations, truncations, _ = env.step(actions)
        
        # Check if episode is done
        dones = [terminations[agent] or truncations[agent] for agent in agents]
        done = any(dones)
        
        # Prepare data for buffer (convert to NumPy once)
        rewards_array = np.array([rewards[agent] for agent in agents], dtype=np.float32)
        next_states_list = [np.array(next_observations[agent], dtype=np.float32) for agent in agents]
        # we care about the termination of the episode
        terminations_array = np.array([terminations[agent] for agent in agents], dtype=np.uint8)
        
        # Store experience in replay buffer
        buffer.add(
            states=states_list,
            actions=actions_list,
            rewards=rewards_array,
            next_states=next_states_list,
            dones=terminations_array
        )
        
        # Update observations and rewards
        observations = next_observations
        episode_rewards += np.array(list(rewards.values()))         
        
        # Learn if enough samples are available in memory
        if global_step > args.warmup_steps and global_step % args.update_every == 0:
            for i in range(len(agents)):
                experiences = buffer.sample()  # Now returns pre-combined states
                critic_loss, actor_loss = maddpg.learn(experiences, i)
                
                # Log losses to TensorBoard
                logger.add_scalar(f'{agents[i]}/critic_loss', critic_loss, global_step)
                logger.add_scalar(f'{agents[i]}/actor_loss', actor_loss, global_step)
                
            maddpg.update_targets()
        
        # Update noise scale based on iteration number
        if global_step > args.warmup_steps and args.use_noise_decay:
            noise_scale = max(
                args.min_noise,
                noise_scale - noise_decay
            )
        
        # Handle episode end
        if done or (global_step % args.max_steps == 0):  # Reset after max_steps if not done
            for i, reward in enumerate(episode_rewards):
                agent_rewards[i].append(reward)
                logger.add_scalar(f"{agents[i]}/episode_reward", reward, global_step)
            logger.add_scalar('train/total_reward', np.sum(episode_rewards), global_step)
            logger.add_scalar(f"noise/scale", noise_scale, global_step)
            observations, _ = env.reset()
            episode_rewards = np.zeros(len(agents))
        
        # Evaluate and save
        if global_step % args.eval_interval == 0 or global_step == args.total_timesteps:
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
    np.save(os.path.join(logger.dir_name, "agent_rewards.npy"), agent_rewards)
    
    # Close environment and TensorBoard writer
    env.close()
    env_evaluate.close()
    logger.close()
    
    # Return both the agent rewards and the experiment name
    return agent_rewards, experiment_name

if __name__ == "__main__":
    args = parse_args()
    train(args)
