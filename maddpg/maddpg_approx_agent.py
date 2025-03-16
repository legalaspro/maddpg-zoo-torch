"""
Deep Deterministic Policy Gradient (DDPG) Agent Implementation
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from .networks import Actor, Critic, ApproxActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPGApproxAgent:
    """
    MADDPG Agent with Approximate Other Agents' Policies
    """
    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64), 
                 actor_lr=1e-4, critic_lr=1e-3, tau=1e-3,
                 agent_idx=None, action_low=None, action_high=None):
        """
        Initialize a MADDPGApprox agent.
        
        Args:
            state_sizes (list): List of dimensions of the state space for each agent
            action_sizes (list): List of dimensions of the action space for each agent
            hidden_sizes (tuple): Sizes of hidden layers for networks
            actor_lr (float): Learning rate for the actor
            critic_lr (float): Learning rate for the critic (not used here, kept for compatibility)
            tau (float): Soft update parameter
            agent_idx (int): Index of the current agent
            action_low (float or array): Lower bound of the action space (default: -1.0)
            action_high (float or array): Upper bound of the action space (default: 1.0)
        """
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.tau = tau
        self.agent_idx = agent_idx
        self.num_agents = len(state_sizes)

        total_state_size = sum(state_sizes)
        total_action_size = sum(action_sizes)
        
        # Set action bounds
        self.action_low = -1.0 if action_low is None else action_low
        self.action_high = 1.0 if action_high is None else action_high
        self.action_range = self.action_high - self.action_low
        
        # Actor Networks (Local and Target)
        self.actor = Actor(state_sizes[self.agent_idx], action_sizes[self.agent_idx], hidden_sizes, 
                          action_low=self.action_low, 
                          action_high=self.action_high).to(device)
        self.actor_target = Actor(state_sizes[self.agent_idx], action_sizes[self.agent_idx], hidden_sizes,
                                 action_low=self.action_low, 
                                 action_high=self.action_high).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic Networks (Local and Target) if using centralized critic            
        self.critic = Critic(total_state_size, total_action_size, hidden_sizes).to(device)
        self.critic_target = Critic(total_state_size, total_action_size, hidden_sizes).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize target networks with local network weights
        self.hard_update(self.critic_target, self.critic)
        self.hard_update(self.actor_target, self.actor)

        self.approx_actors = []
        self.approx_actor_targets = []
        self.approx_actor_optimizers = []

        for i in range(self.num_agents):
            if i != self.agent_idx:
                self.approx_actors.append(ApproxActor(state_sizes[i], action_sizes[i], hidden_sizes, 
                                                     action_low=self.action_low, 
                                                     action_high=self.action_high).to(device))
                self.approx_actor_targets.append(ApproxActor(state_sizes[i], action_sizes[i], hidden_sizes, 
                                                     action_low=self.action_low, 
                                                     action_high=self.action_high).to(device))
                self.approx_actor_optimizers.append(optim.Adam(self.approx_actors[i].parameters(), lr=actor_lr))
                self.hard_update(self.approx_actor_targets[i], self.approx_actors[i])
            else:
                self.approx_actors.append(None)
                self.approx_actor_targets.append(None)
                self.approx_actor_optimizers.append(None)
               
        
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
    
    def _learn_approx_policies(self, states, actions, gamma=0.99):
        """
        Update approximate policies of other agents using given batch of experience tuples.
        """

        for i in range(self.num_agents):
            if i != self.agent_idx:
                approx_actor = self.approx_actors[i]
                approx_actor_target = self.approx_actor_targets[i]
                approx_actor_optimizer = self.approx_actor_optimizers[i]
            
                log_prob, entropy = approx_actor.evaluate_actions(states[i], actions[i])

                # Calculate the loss with the critic values
                loss = (-log_prob - 1e-3 * entropy).mean()

                # Minimize the loss
                approx_actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(approx_actor.parameters(), 1.0)
                approx_actor_optimizer.step()

                self.soft_update(approx_actor_target, approx_actor)

    def _act_target(self, states):
        """
        Get actions from all agents based on target policies.
        """
        actions = []
        for i in range(self.num_agents):
            if i != self.agent_idx:
                action, _, _ = self.approx_actor_targets[i].sample(states[i])
                actions.append(action)
            else:
                actions.append(self.actor_target(states[i]))

        return actions

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
        states, actions, rewards, next_states, dones, states_full, next_states_full, actions_full = experiences
        
        # ---------------------------- update approximate policies ---------------------------- #
        self._learn_approx_policies(states, actions, gamma)

        agent_rewards = rewards[self.agent_idx]
        agent_dones = dones[self.agent_idx]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_actions_list = self._act_target(next_states)
            next_actions_full = torch.cat(next_actions_list, dim=1)
            Q_targets_next = self.critic_target(next_states_full, next_actions_full)
            # Compute Q targets for current states (y_i)
            Q_targets = agent_rewards + (gamma * Q_targets_next * (1 - agent_dones))
        
        # Compute critic loss
        Q_expected = self.critic(states_full, actions_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #

        # Compute actor loss
        actions_pred = []
        for i in range(self.num_agents):
            if i == self.agent_idx:
                actions_pred.append(self.actor(states[i]))
            else:
                actions_pred.append(actions[i].detach())
        
        actions_full_pred = torch.cat(actions_pred, dim=1)

        actor_loss = -self.critic(states_full, actions_full_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
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
    
    