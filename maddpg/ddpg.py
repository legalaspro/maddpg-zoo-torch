"""
Deep Deterministic Policy Gradient (DDPG) Agent Implementation
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from .networks import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    """
    DDPG Agent with Actor network
    """
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), 
                 actor_lr=1e-4, critic_lr=1e-3, tau=1e-3,
                 centralized=False, total_state_size=None, total_action_size=None,
                 action_low=None, action_high=None):
        """
        Initialize a DDPG agent.
        
        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
            hidden_sizes (tuple): Sizes of hidden layers for networks
            actor_lr (float): Learning rate for the actor
            critic_lr (float): Learning rate for the critic (not used here, kept for compatibility)
            tau (float): Soft update parameter
            centralized (bool): Whether to use centralized critic
            total_state_size (int): Total dimension of all agents' states (for centralized critic)
            total_action_size (int): Total dimension of all agents' actions (for centralized critic)
            action_low (float or array): Lower bound of the action space (default: -1.0)
            action_high (float or array): Upper bound of the action space (default: 1.0)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.centralized = centralized
        
        # Set action bounds
        self.action_low = -1.0 if action_low is None else action_low
        self.action_high = 1.0 if action_high is None else action_high
        self.action_range = self.action_high - self.action_low
        
        # Actor Networks (Local and Target)
        self.actor = Actor(state_size, action_size, hidden_sizes, 
                          action_low=self.action_low, 
                          action_high=self.action_high).to(device)
        self.actor_target = Actor(state_size, action_size, hidden_sizes,
                                 action_low=self.action_low, 
                                 action_high=self.action_high).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic Networks (Local and Target) if using centralized critic
        if centralized:
            # Centralized critic takes all states and actions
            critic_state_size = total_state_size
            critic_action_size = total_action_size
        else:
            # Decentralized critic takes only this agent's state and action
            critic_state_size = state_size
            critic_action_size = action_size
            
        self.critic = Critic(critic_state_size, critic_action_size, hidden_sizes).to(device)
        self.critic_target = Critic(critic_state_size, critic_action_size, hidden_sizes).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize target networks with local network weights
        self.hard_update(self.critic_target, self.critic)
        self.hard_update(self.actor_target, self.actor)
        
    def act(self, state, add_noise=True, noise_scale=1.0):
        """
        Returns actions for given state as per current policy.
        
        Args:
            state: Current state
            add_noise (bool): Whether to add noise for exploration
            noise_scale (float): Scale factor for noise
        """
        state = torch.from_numpy(state).float().to(device)
        
        self.actor.eval()
        with torch.no_grad():
            # Get action from network (already scaled to [action_low, action_high])
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        
        if add_noise:
            # Scale noise by action range and noise_scale
            scaled_noise = np.random.normal(0, noise_scale * self.action_range, size=action.shape)
            action += scaled_noise
            
        # Clip to [action_low, action_high] range
        return np.clip(action, self.action_low, self.action_high)
    
    def act_target(self, state):
        """
        Returns actions for given state as per current target policy.
        Keeps gradients for learning.
        
        Args:
            state: Current state (tensor)
            
        Returns:
            action: Action from target policy (tensor)
        """
        # Assume state is already a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
            
        # Return tensor directly (with gradients)
        return self.actor_target(state)
    
    def learn(self, experiences, gamma=0.99):
        """
        Update policy and value parameters using given batch of experience tuples.
        Implements the standard DDPG algorithm without centralized critics.
        
        Args:
            experiences (tuple): (states, actions, rewards, next_states, dones)
            gamma (float): Discount factor
            
        Returns:
            critic_loss (float): Loss of the critic network
            actor_loss (float): Loss of the actor network
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()
        
        # ---------------------------- update target networks ---------------------------- #
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)
        
        return critic_loss.item(), actor_loss.item()
    
    def hard_update(self, target_model, source_model):
        """
        Hard update model parameters.
        θ_target = θ_source
        
        Args:
            target_model: Model with weights to copy to
            source_model: Model with weights to copy from
        """
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(source_param.data)
        
    def save(self, path, agent_idx):
        """Save agent models"""
        torch.save(self.actor.state_dict(), f"{path}/actor_{agent_idx}.pth")
        if self.centralized:
            torch.save(self.critic.state_dict(), f"{path}/critic_{agent_idx}.pth")
        
    def load(self, path, agent_idx):
        """Load agent models"""
        self.actor.load_state_dict(torch.load(f"{path}/actor_{agent_idx}.pth"))
        self.actor_target.load_state_dict(torch.load(f"{path}/actor_{agent_idx}.pth"))
        
        if self.centralized:
            critic_path = f"{path}/critic_{agent_idx}.pth"
            if os.path.exists(critic_path):
                self.critic.load_state_dict(torch.load(critic_path))
                self.critic_target.load_state_dict(torch.load(critic_path)) 
    
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