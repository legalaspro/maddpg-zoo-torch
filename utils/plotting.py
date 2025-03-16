import os
import numpy as np
import matplotlib.pyplot as plt


def plot_rewards_single_env(agents, agent_rewards, dir, env_name=None, window_size=10, 
                            target_score=None, figsize=(12, 8)):
    """
    Plot rewards for each agent from single environment training.
    
    Args:
        agents: List of agent names
        agent_rewards: List of reward data for each agent (agent_rewards[agent_idx][episode_idx] = reward)
        dir: Directory to save the plots
        window_size: Size of window for running average
        target_score: Optional target score to display as horizontal line
        figsize: Figure dimensions
    """
    num_agents = len(agents)

    # Create title prefix based on environment name
    env_title = f"{env_name} - " if env_name else ""
    
    # Create a plot for each agent
    for i, agent in enumerate(agents):
        fig, ax = plt.subplots(figsize=figsize)
        
        episodes = np.arange(1, len(agent_rewards[i]) + 1)
        rewards = np.array(agent_rewards[i])
        
        # Calculate running statistics (mean, min, max)
        avg_rewards = []
        min_rewards = []
        max_rewards = []
        
        for j in range(len(rewards)):
            if j < window_size:
                window = rewards[:j+1]
            else:
                window = rewards[j-window_size+1:j+1]
            
            avg_rewards.append(np.mean(window))
            min_rewards.append(np.min(window))
            max_rewards.append(np.max(window))
        
        # Convert to numpy arrays
        avg_rewards = np.array(avg_rewards)
        min_rewards = np.array(min_rewards)
        max_rewards = np.array(max_rewards)
        
        # Plot the running average with min/max range
        ax.plot(episodes, avg_rewards, linewidth=2, color='blue', label=f'{window_size}-Episode Average')
        ax.fill_between(episodes, min_rewards, max_rewards, alpha=0.2, color='blue', 
                        label=f'Min-Max Range (Window={window_size})')
        
        # Plot target line if provided
        if target_score is not None:
            ax.axhline(y=target_score, color='green', linestyle='--', linewidth=2, 
                      label=f'Target Score ({target_score})')
        
        # Customize the plot
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f'{env_title}{agent} Training Rewards', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{agent}_rewards.png"), dpi=300)
        plt.close()
    
    # Create a combined plot with all agents
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each agent's running average
    for i, agent in enumerate(agents):
        episodes = np.arange(1, len(agent_rewards[i]) + 1)
        rewards = np.array(agent_rewards[i])
        
        # Calculate running average
        avg_rewards = []
        for j in range(len(rewards)):
            if j < window_size:
                avg_rewards.append(np.mean(rewards[:j+1]))
            else:
                avg_rewards.append(np.mean(rewards[j-window_size+1:j+1]))
        
        ax.plot(episodes, avg_rewards, linewidth=2, label=f'{agent} (Avg)')
    
    # Calculate and plot total rewards
    total_rewards = np.zeros(len(agent_rewards[0]))
    for agent_idx in range(num_agents):
        total_rewards += np.array(agent_rewards[agent_idx])
    
    # Calculate running statistics for total rewards
    avg_total = []
    min_total = []
    max_total = []
    
    for j in range(len(total_rewards)):
        if j < window_size:
            window = total_rewards[:j+1]
        else:
            window = total_rewards[j-window_size+1:j+1]
        
        avg_total.append(np.mean(window))
        min_total.append(np.min(window))
        max_total.append(np.max(window))
    
    # Plot total rewards with min/max range
    ax.plot(episodes, avg_total, linewidth=3, color='black', label='Total (Avg)')
    ax.fill_between(episodes, min_total, max_total, alpha=0.2, color='gray', 
                    label=f'Total Min-Max Range')
    
    # Plot target line if provided
    if target_score is not None:
        ax.axhline(y=target_score * num_agents, color='green', linestyle='--', linewidth=2, 
                  label=f'Target Total ({target_score * num_agents})')
    
    # Customize the plot
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'{env_title}All Agents Rewards', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "all_agents_rewards.png"), dpi=300)
    plt.close()

def compare_algorithms(env_name, algo_paths, output_dir, window_size=100, figsize=(12, 8)):
    """
    Compare total rewards from different algorithm runs on the same plot.
    
    Args:
        env_name: Name of the environment (for the plot title)
        algo_paths: Dictionary mapping algorithm names to their results paths
                    e.g., {'MADDPG': 'path/to/maddpg/agent_rewards.npy',
                           'MADDPG-Approx': 'path/to/maddpg_approx/agent_rewards.npy'}
        output_dir: Directory to save the plots
        window_size: Size of window for running average
        figsize: Figure dimensions
    """
    plt.figure(figsize=figsize)
    
    # Define a color palette for different algorithms
    colors = {
        'MADDPG': '#0072B2',       # Blue
        'MADDPG-Approx': '#D55E00', # Orange/Red
        'DDPG': '#009E73',         # Green
        'PPO': '#CC79A7',          # Pink
        'SAC': '#56B4E9'           # Light Blue
    }
    
    # Store max episode length to align plots
    max_episodes = 0
    
    # Process and plot each algorithm's data
    for algo_name, results_path in algo_paths.items():
        if not os.path.exists(results_path):
            print(f"Warning: Results file not found for {algo_name} at {results_path}")
            continue
            
        # Load the agent rewards data
        agent_rewards = np.load(results_path, allow_pickle=True)
        
        # Determine if we have single or multi-environment data
        is_multi_env = isinstance(agent_rewards[0][0], (list, np.ndarray)) and len(agent_rewards[0][0]) > 1
        
        # Calculate total rewards across all agents
        if is_multi_env:
            # For multi-env: Extract means from [mean, min, max] format
            total_rewards = np.sum([np.array([r[0] for r in agent_data]) for agent_data in agent_rewards], axis=0)
        else:
            # For single-env: Sum raw rewards
            total_rewards = np.sum([agent_data for agent_data in agent_rewards], axis=0)
        
        # Update max episode count
        max_episodes = max(max_episodes, len(total_rewards))
        
        # Calculate running average
        smoothed_rewards = []
        for i in range(len(total_rewards)):
            if i < window_size:
                smoothed_rewards.append(np.mean(total_rewards[:i+1]))
            else:
                smoothed_rewards.append(np.mean(total_rewards[i-window_size+1:i+1]))
        
        # Get color for this algorithm (default to black if not in the color dict)
        color = colors.get(algo_name, 'black')
        
        # Plot the smoothed rewards
        episodes = np.arange(1, len(total_rewards) + 1)
        plt.plot(episodes, smoothed_rewards, linewidth=2, color=color, label=f'{algo_name}')
        
        # Calculate and plot min-max range for the smoothed rewards
        min_rewards = []
        max_rewards = []
        for i in range(len(total_rewards)):
            if i < window_size:
                window = total_rewards[:i+1]
            else:
                window = total_rewards[i-window_size+1:i+1]
            min_rewards.append(np.min(window))
            max_rewards.append(np.max(window))
        
        plt.fill_between(episodes, min_rewards, max_rewards, alpha=0.2, color=color)
    
    # Add plot details
    plt.title(f'{env_name} - Algorithm Comparison', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set consistent x-axis limits
    plt.xlim([0, max_episodes])
    
    # Add horizontal lines for performance benchmarks (customize these for your environment)
    if env_name == 'simple_spread_v3':
        plt.axhline(y=-60, color='gray', linestyle='--', alpha=0.5, label='Below Average (-60)')
        plt.axhline(y=-40, color='gray', linestyle='--', alpha=0.5, label='Average (-40)')
        plt.axhline(y=-20, color='gray', linestyle='--', alpha=0.5, label='Excellent (-20)')
    elif env_name == 'simple_adversary_v3':
        # Add appropriate benchmarks for simple_adversary
        pass
    
    plt.tight_layout()
    
    # Save the comparison plot
    plt.savefig(os.path.join(output_dir, f"{env_name}_algorithm_comparison.png"), dpi=300)
    plt.show()


def plot_rewards_multi_env(agents, agent_rewards, dir, window_size=10, target_score=None, figsize=(12, 8)):
    """
    Plot rewards for each agent from multi-environment training.
    
    Args:
        agents: List of agent names
        agent_rewards: List of reward data for each agent 
                      (agent_rewards[agent_idx][episode_idx] = [mean, min, max])
        dir: Directory to save the plots
        window_size: Size of window for running average
        target_score: Optional target score to display as horizontal line
        figsize: Figure dimensions
    """
    num_agents = len(agents)
    
    # Create a plot for each agent
    for i, agent in enumerate(agents):
        fig, ax = plt.subplots(figsize=figsize)
        
        episodes = np.arange(1, len(agent_rewards[i]) + 1)
        means = [r[0] for r in agent_rewards[i]]  # Extract means
        mins = [r[1] for r in agent_rewards[i]]   # Extract mins
        maxs = [r[2] for r in agent_rewards[i]]   # Extract maxs
        
        # Plot the min-max range
        ax.fill_between(episodes, mins, maxs, alpha=0.2, color='blue', label='Min-Max Range')
        
        # Plot the mean scores
        ax.plot(episodes, means, 'o-', markersize=2, alpha=0.7, color='blue', label='Episode Mean')
        
        # Calculate running average
        avg_means = []
        for j in range(len(means)):
            if j < window_size:
                avg_means.append(np.mean(means[:j+1]))
            else:
                avg_means.append(np.mean(means[j-window_size+1:j+1]))
        
        # Plot the running average
        ax.plot(episodes, avg_means, linewidth=3, color='red', label=f'{window_size}-Episode Average')
        
        # Plot target line if provided
        if target_score is not None:
            ax.axhline(y=target_score, color='green', linestyle='--', linewidth=2, 
                      label=f'Target Score ({target_score})')
        
        # Customize the plot
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f'{agent} Training Rewards', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{agent}_rewards.png"), dpi=300)
        plt.close()
    
    # Create a combined plot with all agents
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, agent in enumerate(agents):
        episodes = np.arange(1, len(agent_rewards[i]) + 1)
        means = [r[0] for r in agent_rewards[i]]  # Extract means
        ax.plot(episodes, means, linewidth=2, label=f'{agent}')
    
    # Plot total rewards (sum across all agents)
    total_means = []
    for ep_idx in range(len(agent_rewards[0])):
        total_means.append(sum(agent_rewards[agent_idx][ep_idx][0] for agent_idx in range(num_agents)))
    
    ax.plot(episodes, total_means, linewidth=3, color='black', label='Total')
    
    # Plot target line if provided
    if target_score is not None:
        ax.axhline(y=target_score * num_agents, color='green', linestyle='--', linewidth=2, 
                  label=f'Target Total ({target_score * num_agents})')
    
    # Customize the plot
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('All Agents Rewards', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "all_agents_rewards.png"), dpi=300)
    plt.close()
    
    # Create a separate plot for total rewards with min-max range
    fig, ax = plt.subplots(figsize=figsize)
    
    total_means = []
    total_mins = []
    total_maxs = []
    
    for ep_idx in range(len(agent_rewards[0])):
        # Sum across agents
        total_means.append(sum(agent_rewards[agent_idx][ep_idx][0] for agent_idx in range(num_agents)))
        total_mins.append(sum(agent_rewards[agent_idx][ep_idx][1] for agent_idx in range(num_agents)))
        total_maxs.append(sum(agent_rewards[agent_idx][ep_idx][2] for agent_idx in range(num_agents)))
    
    # Plot the min-max range
    ax.fill_between(episodes, total_mins, total_maxs, alpha=0.2, color='blue', label='Min-Max Range')
    
    # Plot the mean scores
    ax.plot(episodes, total_means, 'o-', markersize=2, alpha=0.7, color='blue', label='Total Mean')
    
    # Calculate running average
    avg_means = []
    for j in range(len(total_means)):
        if j < window_size:
            avg_means.append(np.mean(total_means[:j+1]))
        else:
            avg_means.append(np.mean(total_means[j-window_size+1:j+1]))
    
    # Plot the running average
    ax.plot(episodes, avg_means, linewidth=3, color='red', label=f'{window_size}-Episode Average')
    
    # Plot target line if provided
    if target_score is not None:
        ax.axhline(y=target_score * num_agents, color='green', linestyle='--', linewidth=2, 
                  label=f'Target Total ({target_score * num_agents})')
    
    # Customize the plot
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Total Rewards (All Agents)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "total_rewards.png"), dpi=300)
    plt.close()

