"""
Environment utilities for MARL training
"""
import numpy as np
import supersuit as ss
from pettingzoo.mpe import (
    simple_adversary_v3,
    simple_crypto_v3,
    simple_push_v3,
    simple_reference_v3,
    simple_speaker_listener_v4,
    simple_spread_v3,
    simple_tag_v3,
    simple_v3,
    simple_world_comm_v3,
)

from utils.utils import needs_padding

# Dictionary mapping environment names to their modules
ENV_MAP = {
    "simple_adversary_v3": simple_adversary_v3,
    "simple_crypto_v3": simple_crypto_v3,
    "simple_push_v3": simple_push_v3,
    "simple_reference_v3": simple_reference_v3,
    "simple_speaker_listener_v4": simple_speaker_listener_v4,
    "simple_spread_v3": simple_spread_v3,
    "simple_tag_v3": simple_tag_v3,
    "simple_v3": simple_v3,
    "simple_world_comm_v3": simple_world_comm_v3,
}

def make_env(env_name, max_cycles=25):
    """
    Create a PettingZoo MPE environment.
    
    Args:
        env_name: Name of the environment from the available MPE environments
        max_cycles: Maximum steps per episode
    Returns:
        env: Vectorized environment
    """
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Available environments: {list(ENV_MAP.keys())}")
    
    # Create the base environment
    env = ENV_MAP[env_name].parallel_env(max_cycles=max_cycles, continuous_actions=True, render_mode="rgb_array")
    
    # Apply action space padding if needed
    env = ss.pad_action_space_v0(env)
    
    # Apply observation padding
    env = ss.pad_observations_v0(env)
    
    # Convert to vector env
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    return env

def get_env_info(env_name, max_steps=25, seed=0, apply_padding=True):
    """
    Create a test environment to get agent information.
    
    Args:
        env_name: Name of the environment from the available MPE environments
        max_steps: Maximum steps per episode
        seed: Random seed
        apply_padding: Whether to apply padding to action and observation spaces
        
    Returns:
        agents: List of agent names
        num_agents: Number of agents
        action_sizes: List of action sizes for each agent (padded if apply_padding=True, original otherwise)
        action_low: Lower bound of action space
        action_high: Upper bound of action space
        state_sizes: List of state sizes for each agent (padded if apply_padding=True, original otherwise)
    """
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Available environments: {list(ENV_MAP.keys())}")
    
    print(f"Creating test environment '{env_name}' to get agent information...")
    
    # Create the base environment
    env_single = ENV_MAP[env_name].parallel_env(max_cycles=max_steps, continuous_actions=True)
    
    # Get original sizes before padding
    env_single.reset(seed=seed)
    agents = env_single.agents
    num_agents = len(agents)
    original_action_sizes = [env_single.action_space(agent).shape[0] for agent in agents]
    original_obs_sizes = [env_single.observation_space(agent).shape[0] for agent in agents]
    
    # Always log original sizes for reference
    print(f"Original action sizes: {original_action_sizes}")
    print(f"Original observation sizes: {original_obs_sizes}")
    
    # Show action bounds for all agents
    print("Action bounds for each agent:")
    action_lows = [env_single.action_space(agent).low[0].item() for agent in agents]
    action_highs = [env_single.action_space(agent).high[0].item() for agent in agents]
    print(f"  {agents}: [{action_lows}, {action_highs}]")
    
    if len(set(action_lows)) == 1 and len(set(action_highs)) == 1:
        print("All agents have the same action bounds.")
    else:
        print("Warning: Agents have different action bounds.")
    
    # Store action bounds (using first agent as representative)
    sample_agent = agents[0]
    action_low = action_lows[0]
    action_high = action_highs[0]
    print(f"Using action bounds from {sample_agent}: [{action_low}, {action_high}]")
    
    # If padding is not needed, return original sizes
    if not apply_padding:
        print(f"Using original (unpadded) sizes for environment: {env_name}")
        print(f"Environment: {env_name}")
        print(f"Number of agents: {num_agents}")
        print(f"Agent names: {agents}")
        env_single.close()
        return agents, num_agents, original_action_sizes, action_low, action_high, original_obs_sizes
    
    # Apply action space padding if needed
    if needs_padding(original_action_sizes):
        print(f"Applying action space padding for {env_name}")
        env_single = ss.pad_action_space_v0(env_single)
    
    # Apply observation padding if needed (always apply for consistency)
    if  needs_padding(original_obs_sizes):
        print(f"Applying observation padding for {env_name}")
        env_single = ss.pad_observations_v0(env_single)
    
    # Reset the environment to access agents after padding
    observations, _ = env_single.reset(seed=seed)
    agents = env_single.agents  # Agents might have changed after padding
    
    # Get padded sizes
    padded_action_sizes = [env_single.action_space(agent).shape[0] for agent in agents]
    padded_state_sizes = [env_single.observation_space(agent).shape[0] for agent in agents]
    print(f"Padded action sizes: {padded_action_sizes}")
    print(f"Padded observation sizes: {padded_state_sizes}")
    print(f"Environment: {env_name}")
    print(f"Number of agents: {num_agents}")
    print(f"Agent names: {agents}")
    
    # Close the single environment
    env_single.close()
    
    return agents, num_agents, padded_action_sizes, action_low, action_high, padded_state_sizes

def create_parallel_env(env_name, max_steps=25, num_envs=4):
    """
    Create parallel environments using SuperSuit.
    
    Args:
        env_name: Name of the environment from the available MPE environments
        max_steps: Maximum steps per episode
        num_envs: Number of parallel environments
        
    Returns:
        env: Vectorized environment with multiple parallel instances
    """
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Available environments: {list(ENV_MAP.keys())}")
    
    print(f"Creating {num_envs} parallel '{env_name}' environments using SuperSuit...")
    
    # Use SuperSuit's concat_vec_envs_v1 to create parallel environments
    env = ss.concat_vec_envs_v1(
        make_env(env_name=env_name, max_cycles=max_steps),
        num_vec_envs=num_envs,
        num_cpus=num_envs,  # Use one CPU per environment
        base_class="gymnasium"
    )
    print(f"Vectorized environment created successfully with {num_envs} parallel environments")
    
    return env 

def create_single_env(env_name, max_steps=25, render_mode="rgb_array", apply_padding=True):
    """
    Create a single environment without padding for testing or visualization.
    
    Args:
        env_name: Name of the environment from the available MPE environments
        max_steps: Maximum steps per episode
        render_mode: Render mode for visualization
        
    Returns:
        env: Single environment instance
    """
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Available environments: {list(ENV_MAP.keys())}")
    
    print(f"Creating single '{env_name}' environment...")
    
    # Create the base environment
    env = ENV_MAP[env_name].parallel_env(max_cycles=max_steps, continuous_actions=True, render_mode=render_mode)
    
    if apply_padding:
        env = ss.pad_action_space_v0(env)
        env = ss.pad_observations_v0(env)

    return env 