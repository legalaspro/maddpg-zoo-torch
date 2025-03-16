import numpy as np
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Replay Buffer for MADDPG with per-agent storage in separate arrays."""
    
    def __init__(self, buffer_size, batch_size, agents, state_sizes, action_sizes):
        """
        Initialize the ReplayBuffer with per-agent storage.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            batch_size (int): Size of each training batch
            agents (list): List of agent IDs (e.g., from env.agents)
            state_sizes (list): List of state sizes per agent (e.g., [14, 10, 10])
            action_sizes (list): List of action sizes per agent (e.g., [5, 5, 5])
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.agents = agents
        self.num_agents = len(agents)
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        
        # Initialize per-agent buffers as separate arrays
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.next_states_buffer = []
        self.dones_buffer = []
        
        # Create separate arrays for each agent
        for i in range(self.num_agents):
            self.states_buffer.append(np.zeros((buffer_size, state_sizes[i]), dtype=np.float32))
            self.actions_buffer.append(np.zeros((buffer_size, action_sizes[i]), dtype=np.float32))
            # Store rewards and dones as scalars, not arrays with extra dimension
            self.rewards_buffer.append(np.zeros(buffer_size, dtype=np.float32))
            self.next_states_buffer.append(np.zeros((buffer_size, state_sizes[i]), dtype=np.float32))
            self.dones_buffer.append(np.zeros(buffer_size, dtype=np.float32))
        
        self.position = 0
        self.size = 0
    
    def add(self, states, actions, rewards, next_states, dones):
        """
        Add a new experience to the buffer.
        
        Args:
            states (list): List of states per agent (variable sizes)
            actions (list): List of actions per agent (variable sizes)
            rewards (list or np.ndarray): Rewards per agent [num_agents]
            next_states (list): List of next states per agent (variable sizes)
            dones (list or np.ndarray): Done flags per agent [num_agents]
        """
        
        # Store experience for each agent
        for i in range(self.num_agents):
            self.states_buffer[i][self.position] = states[i]
            self.actions_buffer[i][self.position] = actions[i]
            # Store reward and done as scalars
            self.rewards_buffer[i][self.position] = rewards[i]
            self.next_states_buffer[i][self.position] = next_states[i]
            self.dones_buffer[i][self.position] = dones[i]
        
        # Update position and size
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self):
        """
        Sample a batch of experiences from the buffer.
        
        Returns:
            tuple: (states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, 
                   states_full, next_states_full, actions_full)
                  Each is a tensor with appropriate shape for MADDPG training
        """
        # Sample indices
        indices = np.random.choice(self.size, self.batch_size, replace=False)
        
        # Initialize tensors for each agent
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        # Collect experiences for each agent
        for i in range(self.num_agents):
            # Get data for this agent
            agent_states = self.states_buffer[i][indices]  # [batch_size, state_size_i]  
            agent_actions = self.actions_buffer[i][indices]  # [batch_size, action_size_i]
            agent_rewards = self.rewards_buffer[i][indices]  # [batch_size]
            agent_next_states = self.next_states_buffer[i][indices]  # [batch_size, state_size_i]
            agent_dones = self.dones_buffer[i][indices]  # [batch_size]
            
            # Convert to tensors
            states_batch.append(torch.tensor(agent_states).float().to(device))  # [agent_idx, batch_size, state_size_i]
            actions_batch.append(torch.tensor(agent_actions).float().to(device))  # [agent_idx, batch_size, action_size_i]
            rewards_batch.append(torch.tensor(agent_rewards).float().to(device))  # [agent_idx, batch_size]
            next_states_batch.append(torch.tensor(agent_next_states).float().to(device))  # [agent_idx, batch_size, state_size_i]
            dones_batch.append(torch.tensor(agent_dones).float().to(device))  # [agent_idx, batch_size]
        
        # Stack rewards and dones directly without squeeze
        rewards_batch = torch.stack(rewards_batch).unsqueeze(-1)  # [num_agents, batch_size, 1]
        dones_batch = torch.stack(dones_batch).unsqueeze(-1)  # [num_agents, batch_size, 1]
        
        # Create full state and action tensors for centralized critic
        states_full = torch.cat(states_batch, dim=-1)  # [batch_size, sum(state_sizes)]
        next_states_full = torch.cat(next_states_batch, dim=-1)  # [batch_size, sum(state_sizes)]
        actions_full = torch.cat(actions_batch, dim=-1)  # [batch_size, sum(action_sizes)]
        
        return (states_batch, actions_batch, rewards_batch, next_states_batch, 
                dones_batch, states_full, next_states_full, actions_full)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size