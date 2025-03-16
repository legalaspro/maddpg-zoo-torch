"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with Approximate Other Agents' Policies Implementation
"""
import torch
import torch.nn.functional as F
import numpy as np
from .maddpg_approx_agent import MADDPGApproxAgent
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPGApprox:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with Approximate Other Agents' Policies implementation
    """
    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64), 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-3, 
                 action_low=-1.0, action_high=1.0):
        """
        Initialize a MADDPGApprox agent.
        
        Args:
            state_sizes (list): List of state sizes for each agent
            action_sizes (list): List of action sizes for each agent
            hidden_sizes (tuple): Sizes of hidden layers for networks
            actor_lr (float): Learning rate for the actor
            critic_lr (float): Learning rate for the critic
            gamma (float): Discount factor
            tau (float): Soft update parameter
            action_low (float or array): Lower bound of the action space (default: -1.0)
            action_high (float or array): Upper bound of the action space (default: 1.0)
        """
        self.num_agents = len(state_sizes)
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high
        
        # Create agents
        self.agents = []
        for i in range(self.num_agents):
            agent = MADDPGApproxAgent(
                state_sizes, 
                action_sizes, 
                hidden_sizes=hidden_sizes,
                actor_lr=actor_lr, 
                critic_lr=critic_lr, 
                tau=tau,
                agent_idx=i,
                action_low=action_low,
                action_high=action_high
            )
            self.agents.append(agent)
        
    def act(self, states, add_noise=True, noise_scale=0.0):
        """
        Get actions from all agents based on current policy.
        
        Args:
            states (list): List of states for each agent
            add_noise (bool): Whether to add noise for exploration
            noise_scale (float): Scale factor for noise
        """
        actions = [agent.act(state, add_noise, noise_scale) 
                  for agent, state in zip(self.agents, states)]
        return actions
        
    def learn(self, experiences, agent_idx):
        """
        Update policy and value parameters for a specific agent using given batch of experience tuples.
        
        Args:
            experiences (tuple): (states, actions, rewards, next_states, dones, states_full, next_states_full, actions_full)
            agent_idx (int): Index of the agent to update
            
        Returns:
            critic_loss (float): Loss of the critic network
            actor_loss (float): Loss of the actor network
        """
        current_agent = self.agents[agent_idx]
        
        critic_loss, actor_loss = current_agent.learn(experiences, self.gamma)
        
        return critic_loss, actor_loss
    
    def update_targets(self):
        """
        Soft update target networks for all agents.
        This should be called after all agents have been updated.
        """
        # print("\nUpdating target networks:")
        for i, agent in enumerate(self.agents):
            # Perform soft update
            self.soft_update(agent.actor_target, agent.actor)
            self.soft_update(agent.critic_target, agent.critic)
    
    def soft_update(self, target, source):
        """
        Soft update model parameters.
        θ_target = τ*θ_source + (1 - τ)*θ_target
        
        Args:
            target: Model with weights to update
            source: Model with weights to copy from
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data) 

    def save(self, path):
        """
        Save all agent models to a single file.
        
        Args:
            path (str): Path to save the models
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create a dictionary to store all models
        models_dict = {}
        
        for i, agent in enumerate(self.agents):
            # Save actor and critic models
            models_dict[f'agent_{i}_actor'] = agent.actor.state_dict()
            models_dict[f'agent_{i}_critic'] = agent.critic.state_dict()
            for j in range(agent.num_agents):
                if agent.approx_actors[j] is not None:
                    models_dict[f'agent_{i}_approx_actor_{j}'] = agent.approx_actors[j].state_dict()
        
        # Save all models to a single file
        torch.save(models_dict, path)
        print(f"Models saved to {path}")
    
    def load(self, path):
        """
        Load all agent models from a single file.
        
        Args:
            path (str): Path to load the models from
        """
        if not os.path.exists(path):
            print(f"Warning: No model file found at {path}")
            return
            
        # Load the dictionary containing all models
        models_dict = torch.load(path, weights_only=False)
        
        for i, agent in enumerate(self.agents):
            # Load actor model
            actor_key = f'agent_{i}_actor'
            if actor_key in models_dict:
                agent.actor.load_state_dict(models_dict[actor_key])
                agent.actor_target.load_state_dict(models_dict[actor_key])
                print(f"Loaded actor model for agent {i}")
            
            # Load critic model
            critic_key = f'agent_{i}_critic'
            if critic_key in models_dict:
                agent.critic.load_state_dict(models_dict[critic_key])
                agent.critic_target.load_state_dict(models_dict[critic_key])
                print(f"Loaded critic model for agent {i}")

            # Load approx actor model
            for j in range(agent.num_agents):
                if agent.approx_actors[j] is not None:
                    approx_actor_key = f'agent_{i}_approx_actor_{j}'
                    if approx_actor_key in models_dict:
                        agent.approx_actors[j].load_state_dict(models_dict[approx_actor_key])
                        print(f"Loaded approx actor model for agent {i} and agent {j}")

        print(f"All models loaded from {path}") 