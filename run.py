"""
Script to run trained agents in various environments
"""
import torch
import numpy as np
import argparse
import os
import imageio
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from maddpg import MADDPG, MADDPGApprox
from utils.env import create_single_env, ENV_MAP, get_env_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                       default="runs/simple_spread_v3/maddpg/single_b256_usteps15_g0.95_t0.01_alr0.001_clr0.002_n0.3_minn0.01_h64,64_20250314_233402/model.pt",
                       help="Path to the trained model file")
    parser.add_argument("--env-name", type=str, default="simple_spread_v3", 
                       choices=list(ENV_MAP.keys()),
                       help="Name of the environment to use")
    parser.add_argument("--algo", type=str, default="MADDPG", choices=["MADDPG", "MADDPGApprox"],
                       help="Algorithm to use")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum steps per episode")
    parser.add_argument("--output-dir", type=str, default="./gifs", help="Directory to save outputs")
    parser.add_argument("--is-parallel", action="store_true", help="Parallel environment")
    parser.add_argument("--create-gif", action="store_true", default=True, 
                       help="Create GIF of episodes")
    parser.add_argument("--episode-separator", type=float, default=1.0, 
                       help="Duration in seconds for the black frame between episodes")
    return parser.parse_args()

def add_text_to_frame(frame, text):
    """Add simple text to a frame using PIL."""
    # Convert numpy array to PIL Image
    img = Image.fromarray(frame)
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Try to use a standard font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Add text at the top left corner
    position = (10, 10) 
    
    # Draw text with a black outline for better visibility
    draw.text(position, text, font=font, fill=(0, 0, 0))
    
    # Convert back to numpy array
    return np.array(img)



def run(args):
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.env_name}/{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get environment information
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name,
        max_steps=args.max_steps,
        apply_padding=args.is_parallel  # No padding for single environment
    )

    # Create environment with appropriate render mode
    render_mode = "rgb_array" if args.create_gif else None
    env = create_single_env(
        env_name=args.env_name,
        max_steps=args.max_steps,
        render_mode=render_mode,
        apply_padding=args.is_parallel
    )
    
    # Create MADDPG agent
    if args.algo == "MADDPG":
        maddpg = MADDPG(
            state_sizes=state_sizes,
            action_sizes=action_sizes,
            hidden_sizes=(64, 64),  # Default hidden sizes
            action_low=action_low,
            action_high=action_high
        )
    elif args.algo == "MADDPGApprox":
        maddpg = MADDPGApprox(
            state_sizes=state_sizes,
            action_sizes=action_sizes,
            hidden_sizes=(64, 64),  # Default hidden sizes
            action_low=action_low,
            action_high=action_high
        ) 
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    # Load trained model
    if os.path.exists(args.model_path):
        maddpg.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"No model found at {args.model_path}, using random policies")
    
    # Track episode statistics
    all_episode_rewards = []
    
    # For combined GIF creation
    all_frames = [] if args.create_gif else None
    
    # Run episodes
    for episode in range(1, args.episodes + 1):
        observations, _ = env.reset()
        episode_rewards = np.zeros(len(agents))
        done = False
        step = 0
        
        # For individual episode frames
        episode_frames = [] if args.create_gif else None
        
        while not done and step < args.max_steps:
            # Get states for all agents
            states = [np.array(observations[agent], dtype=np.float32) for agent in agents]
            
            # Get actions for all agents (no noise for evaluation)
            actions_list = maddpg.act(states, add_noise=False)
            
            # Convert actions to dictionary for environment
            actions = {agent: action for agent, action in zip(agents, actions_list)}
            
            # Take a step in the environment
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            
            # Check if episode is done
            dones = [terminations[agent] or truncations[agent] for agent in agents]
            done = any(dones)
            
            # Update observations
            observations = next_observations
            
            # Update rewards
            episode_rewards += np.array(list(rewards.values()))
            
            # Capture frame for GIF if needed
            if args.create_gif:
                frame = env.render()
                
                # Add simple episode and reward text
                text = f"Ep {episode} - R: {np.sum(episode_rewards):.1f}"
                labeled_frame = add_text_to_frame(frame, text)
                
                episode_frames.append(labeled_frame)
                all_frames.append(labeled_frame)
            
            step += 1
        
        # Save episode statistics
        all_episode_rewards.append(episode_rewards)
        
        # Print episode results
        print(f"Episode {episode}, Rewards: {episode_rewards}, Total: {np.sum(episode_rewards)}")
        
        # Add a simple separator between episodes
        if args.create_gif and episode < args.episodes and episode_frames:
            # Get the shape of the frames
            frame_shape = episode_frames[0].shape
            
            # Create a black frame with the same dimensions
            black_frame = np.zeros(frame_shape, dtype=np.uint8)
            
            # Add simple episode text
            next_text = f"Episode {episode+1}"
            next_frame = add_text_to_frame(black_frame, next_text)
            
            # Add separator frames
            separator_frames = int(args.episode_separator * 10)  # 10 frames per second
            for _ in range(separator_frames):
                all_frames.append(next_frame)
    
    # Save combined GIF with all episodes
    if args.create_gif and all_frames:
        combined_gif_path = os.path.join(output_dir, f"{args.algo}_all_episodes.gif")
        try:
            imageio.mimsave(combined_gif_path, all_frames, duration=0.1)  # 100ms per frame
            print(f"Saved combined GIF with all episodes to {combined_gif_path}")
        except Exception as e:
            print(f"Error saving combined GIF: {e}")
    
    # Print summary statistics
    avg_rewards = np.mean(all_episode_rewards, axis=0)
    
    print("\nEvaluation Summary:")
    for i, agent_name in enumerate(agents):
        print(f"{agent_name} average reward: {avg_rewards[i]:.2f}")
    print(f"Average total reward: {np.sum(avg_rewards):.2f}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    args = parse_args()
    run(args) 