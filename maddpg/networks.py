"""
Neural Network architectures for DDPG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(module, output_layer=None, init_w=3e-3):
    """Initialize network weights using standard PyTorch initialization
    
    Args:
        module: PyTorch module to initialize
        output_layer: The output layer that should use uniform initialization
        init_w: Weight initialization range for output layer
    """
    if isinstance(module, nn.Linear):
        if module == output_layer:  # Output layer
            # Use uniform initialization for the final layer
            nn.init.uniform_(module.weight, -init_w, init_w)
            nn.init.uniform_(module.bias, -init_w, init_w)
        else:  # Hidden layers
            # Use Kaiming initialization for ReLU layers
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(module.bias)

def _init_weights_approx(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)  # Stable gain
        nn.init.constant_(module.bias, 0)
    if hasattr(module, 'fc3') and module is module.fc3:  # Final layer
        nn.init.uniform_(module.weight, -3e-3, 3e-3)
        nn.init.constant_(module.bias, 0)


class Actor(nn.Module):
    """Actor (Policy) Model"""
    
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3, 
                 action_low=-1.0, action_high=1.0):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_sizes (tuple): Sizes of hidden layers
            init_w (float): Final layer weight initialization
            action_low (float or array): Lower bound of the action space (default: -1.0)
            action_high (float or array): Upper bound of the action space (default: 1.0)
        """
        super(Actor, self).__init__()
        
        self.action_low = action_low
        self.action_high = action_high
        self.scale = (action_high - action_low) / 2.0
        self.bias = (action_high + action_low) / 2.0
        
        # Build the network
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, self.fc3, init_w))
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output is in range [-1, 1]
        
        # Scale from [-1, 1] to [action_low, action_high]
        return self._scale_action(x)
    
    def _scale_action(self, action):
        """Scale action from [-1, 1] to [action_low, action_high]"""
        return self.scale * action  + self.bias

class Critic(nn.Module):
    """Critic (Value) Model"""
    
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_sizes (tuple): Sizes of hidden layers
            init_w (float): Final layer weight initialization
        """
        super(Critic, self).__init__()
        
        # Build the network - concatenate state and action at the first layer
        self.fc1 = nn.Linear(state_size + action_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, self.fc3, init_w))
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values"""
        # Concatenate state and action at the first layer
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output is Q-value 




# Problem: TanhTransform in TransformedDistribution computes log_prob by inverting 
# the transform: atanh((action - bias) / scale). For your setup (bias=0.5, scale=0.5), 
# actions of 0 or 1 map to atanh(-1) or atanh(1), which are undefined (infinite in 
# theory, NaN in practice due to numerical limits).
# 
# Solution: Clamp the action to [-0.999999, 0.999999] before inverting the transform.
# This avoids the undefined values and the NaN in practice (edge cases).
# 
# Other Solution is adjust the action space definition 
class SafeTanhTransform(torch.distributions.transforms.TanhTransform):
    """Safe Tanh Transform"""

    def _inverse(self, y):
        """Inverse of the TanhTransform"""
        # Clamp to avoid exact -1 or 1
        y = torch.clamp(y, -0.999999, 0.999999)
        return torch.atanh(y)

class ApproxActor(nn.Module):
    """Approximate Actor Network"""

    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3, 
                 action_low=-1.0, action_high=1.0):
        super(ApproxActor, self).__init__()
        
        # Build the network
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size * 2)

        self.action_low = action_low
        self.action_high = action_high
        self.scale = (action_high - action_low) / 2.0
        self.bias = (action_high + action_low) / 2.0

        self.apply(_init_weights_approx)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu, log_std = self.fc3(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def _get_dist(self, mu, log_std):
        """Get the distribution of the action"""
        base_distribution = torch.distributions.Normal(mu, torch.exp(log_std))
        # tanh_transform = torch.distributions.transforms.TanhTransform(cache_size=1)
        tanh_transform = SafeTanhTransform(cache_size=1)
        scale_transform = torch.distributions.transforms.AffineTransform(self.bias, self.scale)
        squashed_and_scaled_dist = torch.distributions.TransformedDistribution(base_distribution, [tanh_transform, scale_transform])
        return squashed_and_scaled_dist, base_distribution

    def sample(self, state, deterministic=False):
        """Sample an action from the actor network"""
        mu, log_std = self.forward(state)
    
        if deterministic:
            action = torch.tanh(mu) * self.scale + self.bias
            return action, None, None
        
        dist, base_dist = self._get_dist(mu, log_std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = base_dist.entropy().sum(-1, keepdim=True) 

        return action, log_prob, entropy
    
    def evaluate_actions(self, state, action):
        """Evaluate the log probability of actions"""
        mu, log_std = self.forward(state)
        
        dist, base_dist = self._get_dist(mu, log_std)

        # Old way of clamping the action
        # action = torch.clamp(action, self.action_low + 1e-6, self.action_high - 1e-6)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = base_dist.entropy().sum(-1, keepdim=True) # 

        return log_prob, entropy