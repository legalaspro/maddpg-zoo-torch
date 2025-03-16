"""
Script to display information about the PettingZoo MPE environment
"""
from pettingzoo.mpe import simple_adversary_v3
import numpy as np

def sample_actions(env):
    """Helper function to sample actions for all agents at once"""
    return {agent: env.action_space(agent).sample() for agent in env.agents}

def print_env_info():
    # Create the environment
    env = simple_adversary_v3.parallel_env(continuous_actions=True)
    
    # Reset the environment
    observations, infos = env.reset()
    
    # Print basic environment information
    print(f"Environment: {env.metadata['name']}")
    print(f"Number of agents: {len(env.agents)}")
    print("\nAgents:")
    for agent in env.agents:
        print(f"  - {agent}")
    
    # Print action space information
    print("\nAction Spaces:")
    for agent in env.agents:
        action_space = env.action_space(agent)
        if hasattr(action_space, 'shape'):
            print(f"  - {agent}: {action_space} (shape: {action_space.shape}, bounds: [{action_space.low}, {action_space.high}])")
        else:
            print(f"  - {agent}: {action_space}")
    
    # Print observation space information
    print("\nObservation Spaces:")
    for agent in env.agents:
        obs_space = env.observation_space(agent)
        if hasattr(obs_space, 'shape'):
            print(f"  - {agent}: {obs_space} (shape: {obs_space.shape})")
        else:
            print(f"  - {agent}: {obs_space}")
    
    # Print sample observation shapes
    print("\nSample Observation Shapes:")
    for agent in observations:
        print(f"  - {agent}: {np.array(observations[agent]).shape}")
    
    # Take a single step with random actions
    print("\n--- Taking a single step with random actions ---")
    
    # Sample actions for all agents at once using the helper function
    actions = sample_actions(env)
    
    print("\nActions taken:")
    for agent, action in actions.items():
        print(f"  - {agent}: {action}")
    
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    
    print("\nRewards received:")
    print(rewards)
    for agent, reward in rewards.items():
        print(f"  - {agent}: {reward}")
    
    print("\nTerminations:")
    for agent, terminated in terminations.items():
        print(f"  - {agent}: {terminated}")
    
    print("\nTruncations:")
    for agent, truncated in truncations.items():
        print(f"  - {agent}: {truncated}")
    
    print("\nNext Observations:")
    for agent, obs in next_obs.items():
        print(f"  - {agent}: {np.array(obs)}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    print_env_info() 