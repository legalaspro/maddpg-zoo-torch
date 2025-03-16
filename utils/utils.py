"""
Utility functions.
"""

import os
import imageio
import matplotlib.pyplot as plt
import numpy as np

def needs_padding(sizes):
    """
    Check if padding is needed by determining if all elements in the list are identical.
    
    Args:
        sizes: List of sizes (action or observation)
        
    Returns:
        bool: True if padding is needed (sizes are not all identical), False otherwise
    """
    return len(set(sizes)) > 1

def save_gif(frames, dir, iteration):
    gif_path = os.path.join(dir, f"batch_{iteration}.gif")
    try:
        # Save GIF with appropriate duration (slower for better viewing)
        imageio.mimsave(gif_path, frames, duration=0.1)  # 100ms per frame
        print(f"Saved GIF for iteration {iteration} to {gif_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")

def evaluate(env, maddpg, logger, record_gif=False, num_eval_episodes=10, end_episode=0, global_step=0):
    """Run evaluation episodes and return average rewards."""
    eval_rewards = [] 
    frames = []  if record_gif else None
    for episode in range(num_eval_episodes):
        observations, _ = env.reset()
        agents = env.agents
        done = False 
        episode_rewards = np.zeros(len(agents))
        while not done:
            states_list = [np.array(observations[agent], dtype=np.float32) for agent in agents]
            actions_list = maddpg.act(states_list, add_noise=False)  # No noise for eval
            actions = {agent: action for agent, action in zip(agents, actions_list)}
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            episode_rewards += np.array(list(rewards.values()))
            dones = [terminations[agent] or truncations[agent] for agent in agents]
            done = any(dones)
            if record_gif and episode == num_eval_episodes - 1: 
                frames.append(env.render())
            observations = next_observations
        eval_rewards.append(episode_rewards) # (num_eval_episodes, num_envs)
    avg_eval_rewards = np.mean(eval_rewards, axis=0) # (num_envs,)

    for i, avg_reward in enumerate(avg_eval_rewards):
        logger.add_scalar(f'{agents[i]}/eval_reward', avg_reward, end_episode if end_episode > 0 else global_step) 

    total_eval_reward = np.sum(eval_rewards) / num_eval_episodes
    logger.add_scalar('eval/total_reward', total_eval_reward, global_step) 
    print(f"Step {global_step}, Eval rewards: {avg_eval_rewards}, Sum: {total_eval_reward}")

    if frames:
        save_gif(frames, logger.dir_name, end_episode if end_episode > 0 else global_step)           
    
    return avg_eval_rewards

def evaluate_ddpg(env, ddpg_agents, logger, record_gif=False, num_eval_episodes=10, end_episode=0, global_step=0):
    """Run evaluation episodes and return average rewards."""
    eval_rewards = [] 
    frames = []  if record_gif else None
    for episode in range(num_eval_episodes):
        observations, _ = env.reset()
        agents = env.agents
        done = False 
        episode_rewards = np.zeros(len(agents))
        while not done:
            states_list = [np.array(observations[agent], dtype=np.float32) for agent in agents]
            actions_list = [ddpg_agents[i].act(states_list[i], add_noise=False) for i in range(len(agents))]
            actions = {agent: action for agent, action in zip(agents, actions_list)}
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            episode_rewards += np.array(list(rewards.values()))
            dones = [terminations[agent] or truncations[agent] for agent in agents]
            done = any(dones)
            if record_gif and episode == num_eval_episodes - 1: 
                frames.append(env.render())
            observations = next_observations
        eval_rewards.append(episode_rewards) # (num_eval_episodes, num_envs)
    avg_eval_rewards = np.mean(eval_rewards, axis=0) # (num_envs,)
    
    for i, avg_reward in enumerate(avg_eval_rewards):
        logger.add_scalar(f'{agents[i]}/eval_reward', avg_reward, end_episode if end_episode > 0 else global_step) 

    total_eval_reward = np.sum(eval_rewards) / num_eval_episodes
    logger.add_scalar('eval/total_reward', total_eval_reward, global_step) 
    print(f"Step {global_step}, Eval rewards: {avg_eval_rewards}, Sum: {total_eval_reward}")    
    
    if frames:
        save_gif(frames, logger.dir_name, end_episode if end_episode > 0 else global_step)           
    
    return avg_eval_rewards
