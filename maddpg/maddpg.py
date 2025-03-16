"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Implementation
"""
import torch
import torch.nn.functional as F
import numpy as np
from .ddpg import DDPGAgent
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation
    """
    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64), 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-3, 
                 action_low=-1.0, action_high=1.0):
        """
        Initialize a MADDPG agent.
        
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
        
        # Calculate total state and action sizes for centralized critic
        self.total_state_size = sum(state_sizes)
        self.total_action_size = sum(action_sizes)
        
        # Create agents
        self.agents = []
        for i in range(self.num_agents):
            agent = DDPGAgent(
                state_sizes[i], 
                action_sizes[i], 
                hidden_sizes=hidden_sizes,
                actor_lr=actor_lr, 
                critic_lr=critic_lr, 
                tau=tau,
                centralized=True,
                total_state_size=self.total_state_size,
                total_action_size=self.total_action_size,
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
        
    def act_target(self, states):
        """
        Get actions from all agents based on target policies.
        
        Args:
            states: States for all agents [batch_size, num_agents, state_size]
            
        Returns:
            actions: List of actions for each agent
        """
        actions = [agent.act_target(state) for agent, state in zip(self.agents, states)]
        
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
        states, actions, rewards, next_states, dones, states_full, next_states_full, actions_full = experiences

        current_agent = self.agents[agent_idx]
        
        # Extract the agent's specific rewards and dones
        agent_rewards = rewards[agent_idx]
        agent_dones = dones[agent_idx]
        
        # Debug: Print shapes of input tensors
        # print(f"\nAgent {agent_idx} Learning:")
        # print(f"  States shape: {states[0].shape}")
        # print(f"  Actions shape: {actions[0].shape}")
        # print(f"  Rewards shape: {agent_rewards.shape}")
        # print(f"  Next states shape: {next_states[0].shape}")
        # print(f"  Dones shape: {agent_dones.shape}")
        # print(f"  States full shape: {states_full.shape}")
        # print(f"  Actions full shape: {actions_full.shape}")
        
        # ---------------------------- update centralized critic ---------------------------- #
        with torch.no_grad():
            # Get predicted next actions for all agents using target networks
            next_actions_list = self.act_target(next_states)
            
            # Debug: Print next actions info
            # print(f"  Next actions list length: {len(next_actions_list)}")
            # print(f"  Next actions[0] shape: {next_actions_list[0].shape}")
        
            # Concatenate next actions for all agents (they're already tensors)
            next_actions_full = torch.cat(next_actions_list, dim=1)
            # print(f"  Next actions full shape: {next_actions_full.shape}")

            # Compute target Q-value
            Q_targets_next = current_agent.critic_target(next_states_full, next_actions_full)
            # print(f"  Q targets next shape: {Q_targets_next.shape}")
            # print(f"  Q targets next mean: {Q_targets_next.mean().item():.6f}")
            
            # Compute Q targets for current states (y_i)
            Q_targets = agent_rewards + (self.gamma * Q_targets_next * (1 - agent_dones))
            # print(f"  Q targets shape: {Q_targets.shape}")
            # print(f"  Q targets mean: {Q_targets.mean().item():.6f}")

        # Compute critic loss
        Q_expected = current_agent.critic(states_full, actions_full)
        # print(f"  Q expected shape: {Q_expected.shape}")
        # print(f"  Q expected mean: {Q_expected.mean().item():.6f}")
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # print(f"  Critic loss: {critic_loss.item():.6f}")

        # Update the critic for the current agent
        current_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(current_agent.critic.parameters(), 1.0)
        current_agent.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        
        # Compute actor loss
        actions_pred = []
        for i, agent in enumerate(self.agents):
            if i == agent_idx:
                actions_pred.append(current_agent.actor(states[i]))
            else: # Detach actions from other agents to prevent gradient flow
                # actions_pred.append(self.agents[i].actor(states[i]).detach())
                actions_pred.append(actions[i].detach())

        actions_full_pred = torch.cat(actions_pred, dim=1)
        # print(f"  Actions full pred shape: {actions_full_pred.shape}")
        
        # Compute actor loss using the agent's critic
        actor_loss = -current_agent.critic(states_full, actions_full_pred).mean()
        # print(f"  Actor loss: {actor_loss.item():.6f}")

        # Update the actor for the current agent
        current_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(current_agent.actor.parameters(), 0.5)
        current_agent.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()
    
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
        
        print(f"All models loaded from {path}") 