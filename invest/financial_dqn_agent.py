"""
Simplified DQN agent for financial trading that leverages angle/RL components.
This bridges the gap between financial environments and gaming DQN infrastructure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
from typing import Dict, List, Tuple, Optional
from collections import deque

# Add angle/RL to path
sys.path.append('/home/ubuntu/code/angle/RL')

# Import components from angle/RL that we can reuse
try:
    from model.device_utils import get_device_manager
except ImportError:
    # Fallback device manager
    class MockDeviceManager:
        def __init__(self, device=None):
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        def to_dev(self, x):
            return x.to(self.device) if hasattr(x, 'to') else x
    
    def get_device_manager(device=None):
        return MockDeviceManager(device)


class DQNNetwork(nn.Module):
    """Simple DQN network for financial data (non-image inputs)."""
    
    def __init__(self, input_dim: int, output_dim: int, use_dueling: bool = True):
        super().__init__()
        self.use_dueling = use_dueling
        
        # Hidden layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        
        if use_dueling:
            # Dueling architecture
            self.value_head = nn.Linear(256, 1)
            self.advantage_head = nn.Linear(256, output_dim)
        else:
            # Standard architecture
            self.q_head = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        if self.use_dueling:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(x)
        
        return q_values


class FinancialDQNAgent:
    """
    DQN Agent specifically designed for financial trading.
    Leverages angle/RL's replay buffer and network components.
    Supports TD3 features: twin Q-networks, tau updates, and policy delay.
    """
    
    def __init__(self, 
                 observation_dim: int,
                 n_actions: int,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 lr: float = 0.0001,
                 batch_size: int = 32,
                 memory_size: int = 10000,
                 target_update_frequency: int = 1000,
                 use_dueling: bool = True,
                 use_prioritized_replay: bool = False,
                 # TD3 options
                 use_twin_networks: bool = False,
                 use_tau_updates: bool = False,
                 tau: float = 0.005,
                 policy_delay: int = 2,
                 device: str = None):
        
        self.devmgr = get_device_manager(device)
        self.device = self.devmgr.device
        
        # Parameters
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # TD3 parameters
        self.use_twin_networks = use_twin_networks
        self.use_tau_updates = use_tau_updates
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_delay_counter = 0
        
        # Networks
        self.network = DQNNetwork(
            input_dim=observation_dim,
            output_dim=n_actions,
            use_dueling=use_dueling
        )
        self.network = self.devmgr.to_dev(self.network)
        
        # Twin network (second Q-network for TD3)
        if use_twin_networks:
            self.network2 = DQNNetwork(
                input_dim=observation_dim,
                output_dim=n_actions,
                use_dueling=use_dueling
            )
            self.network2 = self.devmgr.to_dev(self.network2)
        else:
            self.network2 = None
        
        # Target networks
        self.target_network = DQNNetwork(
            input_dim=observation_dim,
            output_dim=n_actions,
            use_dueling=use_dueling
        )
        self.target_network = self.devmgr.to_dev(self.target_network)
        self.target_network.load_state_dict(self.network.state_dict())
        
        if use_twin_networks:
            self.target_network2 = DQNNetwork(
                input_dim=observation_dim,
                output_dim=n_actions,
                use_dueling=use_dueling
            )
            self.target_network2 = self.devmgr.to_dev(self.target_network2)
            self.target_network2.load_state_dict(self.network2.state_dict())
        else:
            self.target_network2 = None
        
        # Optimizers
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        if use_twin_networks:
            self.optimizer2 = optim.Adam(self.network2.parameters(), lr=lr)
        else:
            self.optimizer2 = None
        
        # Replay buffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            # Use simple deque for now since angle/RL ReplayBuffer expects frames
            self.memory = deque(maxlen=memory_size)
        self.use_prioritized = use_prioritized_replay
        
        # Training stats
        self.total_steps = 0
        self.training_steps = 0
        
        print(f"Financial DQN Agent initialized:")
        print(f"  Observation dim: {observation_dim}")
        print(f"  Actions: {n_actions}")
        print(f"  Memory: {'Prioritized' if use_prioritized_replay else 'Standard'}")
        print(f"  Device: {self.device}")
        if use_twin_networks or use_tau_updates or policy_delay > 1:
            print(f"  TD3 Features:")
            print(f"    Twin networks: {use_twin_networks}")
            print(f"    Tau updates: {use_tau_updates} (tau={tau if use_tau_updates else 'N/A'})")
            print(f"    Policy delay: {policy_delay}")
            print(f"    Dueling: {use_dueling}")
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        For twin networks, uses minimum Q-value for conservative action selection.
        
        Args:
            observation: Current observation
            
        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            obs_tensor = self.devmgr.to_dev(obs_tensor)
            
            q_values = self.network(obs_tensor)
            
            # For twin networks, use minimum Q-value for conservative action selection
            if self.use_twin_networks:
                q_values2 = self.network2(obs_tensor)
                q_values = torch.minimum(q_values, q_values2)
            
            action = q_values.argmax().item()
        
        return action
    
    def get_q_values(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get Q-values for monitoring purposes.
        
        Args:
            observation: Current observation
            
        Returns:
            Dictionary with Q-values from all networks
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            obs_tensor = self.devmgr.to_dev(obs_tensor)
            
            result = {}
            q_values = self.network(obs_tensor)
            result['q1'] = q_values.cpu().numpy()[0]
            
            if self.use_twin_networks:
                q_values2 = self.network2(obs_tensor)
                result['q2'] = q_values2.cpu().numpy()[0]
                result['q_min'] = torch.minimum(q_values, q_values2).cpu().numpy()[0]
            
            return result
    
    def store_experience(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode terminated
        """
        if self.use_prioritized:
            # For prioritized replay, we need to compute TD error
            with torch.no_grad():
                state_tensor = self.devmgr.to_dev(torch.FloatTensor(state).unsqueeze(0))
                next_state_tensor = self.devmgr.to_dev(torch.FloatTensor(next_state).unsqueeze(0))
                
                current_q = self.network(state_tensor)[0, action]
                next_q = self.target_network(next_state_tensor).max()
                td_error = abs(reward + self.gamma * next_q * (1 - done) - current_q)
                
                self.memory.add(state, action, reward, next_state, done, td_error.item())
        else:
            # Simple replay buffer
            self.memory.append((state, action, reward, next_state, done))
        
        self.total_steps += 1
    
    def train(self) -> Optional[Dict[str, float]]:
        """
        Train the network on a batch of experiences.
        Implements TD3 features when enabled.
        
        Returns:
            Dictionary with loss values or None if not enough samples
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        if self.use_prioritized:
            batch, indices, weights = self.memory.sample(self.batch_size)
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            weights = torch.FloatTensor(weights)
        else:
            batch = random.sample(self.memory, self.batch_size)
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            weights = torch.ones(self.batch_size)
        
        # Move to device
        states = self.devmgr.to_dev(states)
        actions = self.devmgr.to_dev(actions)
        rewards = self.devmgr.to_dev(rewards)
        next_states = self.devmgr.to_dev(next_states)
        dones = self.devmgr.to_dev(dones)
        weights = self.devmgr.to_dev(weights)
        
        # Current Q values from both networks
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))
        losses = {}
        
        if self.use_twin_networks:
            current_q_values2 = self.network2(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (from target networks)
        with torch.no_grad():
            if self.use_twin_networks:
                # TD3: Use minimum of twin target networks
                next_q_values1 = self.target_network(next_states).max(1)[0]
                next_q_values2 = self.target_network2(next_states).max(1)[0]
                next_q_values = torch.minimum(next_q_values1, next_q_values2)
            else:
                next_q_values = self.target_network(next_states).max(1)[0]
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute losses
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        
        if self.use_prioritized:
            # Weighted loss for prioritized replay
            loss1 = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
            
            # Update priorities
            new_priorities = td_errors.abs().squeeze().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, new_priorities)
        else:
            loss1 = td_errors.pow(2).mean()
        
        losses['q1_loss'] = loss1.item()
        
        # Optimize first network
        self.optimizer.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()
        
        # Train second network if using twin networks
        if self.use_twin_networks:
            td_errors2 = target_q_values.unsqueeze(1) - current_q_values2
            
            if self.use_prioritized:
                loss2 = (weights.unsqueeze(1) * td_errors2.pow(2)).mean()
            else:
                loss2 = td_errors2.pow(2).mean()
            
            losses['q2_loss'] = loss2.item()
            
            self.optimizer2.zero_grad()
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.network2.parameters(), 10)
            self.optimizer2.step()
        
        self.training_steps += 1
        self.policy_delay_counter += 1
        
        # Update target networks
        if self.policy_delay_counter >= self.policy_delay:
            self.policy_delay_counter = 0
            
            if self.use_tau_updates:
                self.soft_update_target_networks()
            elif self.training_steps % self.target_update_frequency == 0:
                self.update_target_network()
        
        return losses
    
    def update_target_network(self):
        """Update target network with current network weights (hard update)."""
        self.target_network.load_state_dict(self.network.state_dict())
        if self.use_twin_networks:
            self.target_network2.load_state_dict(self.network2.state_dict())
    
    def soft_update_target_networks(self):
        """Soft update target networks using tau parameter."""
        # Update first target network
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update second target network if using twin networks
        if self.use_twin_networks:
            for target_param, param in zip(self.target_network2.parameters(), self.network2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save agent state."""
        save_dict = {
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'training_steps': self.training_steps,
            'use_twin_networks': self.use_twin_networks,
            'use_tau_updates': self.use_tau_updates,
            'tau': self.tau,
            'policy_delay': self.policy_delay,
            'policy_delay_counter': self.policy_delay_counter
        }
        
        if self.use_twin_networks:
            save_dict['network2'] = self.network2.state_dict()
            save_dict['target_network2'] = self.target_network2.state_dict()
            save_dict['optimizer2'] = self.optimizer2.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.training_steps = checkpoint['training_steps']
        
        # Load TD3 parameters if they exist
        if 'policy_delay_counter' in checkpoint:
            self.policy_delay_counter = checkpoint['policy_delay_counter']
        
        # Load twin networks if they exist
        if self.use_twin_networks and 'network2' in checkpoint:
            self.network2.load_state_dict(checkpoint['network2'])
            self.target_network2.load_state_dict(checkpoint['target_network2'])
            self.optimizer2.load_state_dict(checkpoint['optimizer2'])
        
        print(f"Agent loaded from {filepath}")


def create_financial_dqn_agent(observation_dim: int,
                              n_actions: int,
                              **kwargs) -> FinancialDQNAgent:
    """
    Factory function to create financial DQN agent.
    
    Args:
        observation_dim: Dimension of observations
        n_actions: Number of discrete actions
        **kwargs: Additional arguments for FinancialDQNAgent
        
    Returns:
        FinancialDQNAgent instance
    """
    return FinancialDQNAgent(
        observation_dim=observation_dim,
        n_actions=n_actions,
        **kwargs
    )