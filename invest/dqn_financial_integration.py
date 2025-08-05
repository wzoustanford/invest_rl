"""
Complete integration module for running DQN/R2D2 on financial data.
This module provides the interface between angle_rl financial data and angle/RL algorithms.
"""

import sys
import os
import torch
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple

# Add angle/RL repo to path
sys.path.append('/home/ubuntu/code/angle/RL')

# Import from angle/RL (with error handling)
try:
    from model.device_utils import get_device_manager
    from config.AgentConfig import AgentConfig
    ANGLE_RL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from angle/RL: {e}")
    ANGLE_RL_AVAILABLE = False
    
    # Fallback implementations
    class MockDeviceManager:
        def __init__(self, device=None):
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        def to_dev(self, x):
            return x.to(self.device) if hasattr(x, 'to') else x
    
    def get_device_manager(device=None):
        return MockDeviceManager(device)
    
    class AgentConfig:
        def __init__(self, **kwargs):
            self.gamma = kwargs.get('gamma', 0.99)
            self.epsilon_start = kwargs.get('epsilon_start', 1.0)
            self.epsilon_end = kwargs.get('epsilon_end', 0.01)
            self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
            self.lr = kwargs.get('lr', 0.0001)
            self.batch_size = kwargs.get('batch_size', 32)
            self.memory_size = kwargs.get('memory_size', 50000)
            self.update_frequency = kwargs.get('update_frequency', 4)
            self.target_update_frequency = kwargs.get('target_update_frequency', 1000)
            self.device = kwargs.get('device', None)
            self.use_dueling = kwargs.get('use_dueling', True)

class MockDQNAgent:
    """Mock DQN agent for testing when angle/RL is not available"""
    def __init__(self, policy_model):
        self.policy_model = policy_model
        self.training = True
        self.epsilon = 1.0
        
    def select_action(self, obs, exploration=True):
        epsilon = self.epsilon if exploration and self.training else 0.0
        return self.policy_model.select_action(torch.from_numpy(obs), epsilon=epsilon)
    
    def store_experience(self, state, action, reward, next_state, done):
        pass  # Mock implementation
    
    def train(self):
        return 0.0  # Mock loss
    
    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)

# Local imports
from model.dqn_financial_wrapper import FinancialEnvironment, FinancialDQNWrapper
from model.dqn_policy_model import FinancialDQNPolicyModel, create_financial_dqn_policy


class FinancialDQNTrainer:
    """
    Main trainer class that integrates financial data with DQN/R2D2 algorithms.
    """
    
    def __init__(self,
                 data_list_file: str,
                 ticker_hash_file: str,
                 algorithm: str = 'dqn',  # 'dqn' or 'r2d2'
                 feature_aggregation: str = 'mean',
                 max_episode_steps: int = 100,
                 device: str = None,
                 use_dueling: bool = True):
        
        self.data_list_file = data_list_file
        self.ticker_hash_file = ticker_hash_file
        self.algorithm = algorithm
        self.device = device
        
        self.devmgr = get_device_manager(device)
        
        # Create financial environment
        self.env = FinancialDQNWrapper(
            data_list_file=data_list_file,
            ticker_hash_file=ticker_hash_file,
            feature_aggregation=feature_aggregation,
            max_episode_steps=max_episode_steps,
            device=device
        )
        
        # Create financial policy model (replicating original architecture)
        self.policy_model = create_financial_dqn_policy(
            data_list_file=data_list_file,
            ticker_hash_file=ticker_hash_file,
            device=device,
            use_dueling=use_dueling
        )
        
        # Create agent config compatible with angle/RL
        self.agent_config = AgentConfig(
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            lr=0.0001,
            batch_size=32,
            memory_size=50000,
            update_frequency=4,
            target_update_frequency=1000,
            device=device,
            use_dueling=use_dueling
        )
        
        print(f"Financial DQN Trainer initialized:")
        print(f"  Algorithm: {algorithm}")
        print(f"  Environment: {type(self.env).__name__}")
        print(f"  Action space: {self.env.action_space.n}")
        print(f"  Observation dim: {self.env.observation_space}")
        print(f"  Device: {self.devmgr.device}")
    
    def create_dqn_agent(self):
        """Create DQN agent from angle/RL repository"""
        if not ANGLE_RL_AVAILABLE:
            print("angle/RL not available, using mock agent")
            return MockDQNAgent(self.policy_model)
        
        try:
            from model.dqn_agent import DQNAgent
            from model.dqn_network import DQNNetwork
            
            # Create DQN network
            network = DQNNetwork(
                input_dim=self.env.observation_space,
                output_dim=self.env.action_space.n,
                use_dueling=self.agent_config.use_dueling
            )
            
            # Create DQN agent
            agent = DQNAgent(
                network=network,
                config=self.agent_config
            )
            
            return agent
            
        except ImportError as e:
            print(f"Error importing DQN components: {e}")
            print("Using mock agent instead")
            return MockDQNAgent(self.policy_model)
    
    def create_r2d2_agent(self):
        """Create R2D2 agent from angle/RL repository"""
        try:
            # Note: This would need R2D2 implementation in angle/RL
            # For now, fallback to DQN
            print("R2D2 not yet implemented, falling back to DQN")
            return self.create_dqn_agent()
            
        except ImportError as e:
            print(f"Error importing R2D2 components: {e}")
            return None
    
    def train_episode(self, agent, episode_num: int = 0) -> Dict:
        """
        Train agent for one episode on financial data.
        
        Args:
            agent: DQN/R2D2 agent
            episode_num: Episode number
        
        Returns:
            Episode statistics
        """
        # Reset environment
        obs, info = self.env.reset()
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        
        episode_stats = {
            'total_reward': 0.0,
            'steps': 0,
            'final_portfolio_value': 1.0,
            'avg_loss': 0.0,
            'final_sharpe': 0.0
        }
        
        done = False
        while not done:
            # Select action
            if hasattr(agent, 'select_action'):
                action = agent.select_action(obs)
            else:
                # Fallback: use policy model
                action = self.policy_model.select_action(
                    torch.from_numpy(obs), 
                    epsilon=max(0.01, 1.0 - episode_num * 0.001)
                )
            
            # Execute action
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience and train
            if hasattr(agent, 'store_experience'):
                agent.store_experience(obs, action, reward, next_obs, done)
            
            if hasattr(agent, 'train') and steps % 4 == 0:
                loss = agent.train()
                if loss is not None:
                    total_loss += loss
            
            # Update state
            obs = next_obs
            total_reward += reward
            steps += 1
            
            # Update episode stats
            if 'portfolio_value' in step_info:
                episode_stats['final_portfolio_value'] = step_info['portfolio_value']
            if 'sharpe' in step_info:
                episode_stats['final_sharpe'] = step_info['sharpe']
        
        episode_stats.update({
            'total_reward': total_reward,
            'steps': steps,
            'avg_loss': total_loss / max(1, steps // 4)
        })
        
        return episode_stats
    
    def train(self, num_episodes: int = 1000, save_frequency: int = 100):
        """
        Train DQN/R2D2 agent on financial data.
        
        Args:
            num_episodes: Number of episodes to train
            save_frequency: How often to save the model
        """
        # Create agent
        if self.algorithm.lower() == 'r2d2':
            agent = self.create_r2d2_agent()
        else:
            agent = self.create_dqn_agent()
        
        if agent is None:
            print("Failed to create agent. Cannot proceed with training.")
            return
        
        print(f"Starting training for {num_episodes} episodes...")
        
        # Training loop
        episode_rewards = []
        episode_losses = []
        
        for episode in range(num_episodes):
            episode_stats = self.train_episode(agent, episode)
            
            episode_rewards.append(episode_stats['total_reward'])
            episode_losses.append(episode_stats['avg_loss'])
            
            # Print progress
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_loss = np.mean(episode_losses[-50:])
                print(f"Episode {episode:4d}: "
                      f"Avg Reward: {avg_reward:8.3f}, "
                      f"Avg Loss: {avg_loss:8.6f}, "
                      f"Portfolio Value: {episode_stats['final_portfolio_value']:.3f}, "
                      f"Sharpe: {episode_stats['final_sharpe']:.3f}")
            
            # Save model
            if episode % save_frequency == 0 and episode > 0:
                self.save_model(agent, f"financial_dqn_episode_{episode}.pth")
        
        print("Training completed!")
        
        # Save final model
        self.save_model(agent, "financial_dqn_final.pth")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses
        }
    
    def evaluate(self, agent, num_episodes: int = 10) -> Dict:
        """
        Evaluate trained agent.
        
        Args:
            agent: Trained agent
            num_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation statistics
        """
        agent.training = False  # Set to evaluation mode
        
        eval_rewards = []
        eval_portfolio_values = []
        eval_sharpe_ratios = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False
            
            while not done:
                # Select action (no exploration)
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(obs, exploration=False)
                else:
                    action = self.policy_model.select_action(
                        torch.from_numpy(obs), 
                        epsilon=0.0
                    )
                
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            eval_rewards.append(total_reward)
            if 'portfolio_value' in step_info:
                eval_portfolio_values.append(step_info['portfolio_value'])
            if 'sharpe' in step_info:
                eval_sharpe_ratios.append(step_info['sharpe'])
        
        eval_stats = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_portfolio_value': np.mean(eval_portfolio_values),
            'avg_sharpe': np.mean(eval_sharpe_ratios),
            'success_rate': np.mean([pv > 1.0 for pv in eval_portfolio_values])
        }
        
        agent.training = True  # Set back to training mode
        
        return eval_stats
    
    def save_model(self, agent, filename: str):
        """Save trained model"""
        try:
            if hasattr(agent, 'network'):
                torch.save(agent.network.state_dict(), filename)
                print(f"Model saved to {filename}")
            else:
                torch.save(self.policy_model.get_network_state_dict(), filename)
                print(f"Policy model saved to {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, agent, filename: str):
        """Load trained model"""
        try:
            if hasattr(agent, 'network'):
                agent.network.load_state_dict(torch.load(filename, map_location=self.devmgr.device))
                print(f"Model loaded from {filename}")
            else:
                self.policy_model.load_network_state_dict(
                    torch.load(filename, map_location=self.devmgr.device)
                )
                print(f"Policy model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")


def run_financial_dqn_experiment(data_list_file: str,
                                ticker_hash_file: str,
                                num_episodes: int = 1000,
                                algorithm: str = 'dqn',
                                device: str = None):
    """
    Convenience function to run a complete financial DQN experiment.
    
    Args:
        data_list_file: Path to financial data list
        ticker_hash_file: Path to ticker hash file
        num_episodes: Number of training episodes
        algorithm: 'dqn' or 'r2d2'
        device: Device to use
    """
    # Create trainer
    trainer = FinancialDQNTrainer(
        data_list_file=data_list_file,
        ticker_hash_file=ticker_hash_file,
        algorithm=algorithm,
        device=device
    )
    
    # Train
    training_results = trainer.train(num_episodes=num_episodes)
    
    # Evaluate
    agent = trainer.create_dqn_agent()
    if agent:
        trainer.load_model(agent, "financial_dqn_final.pth")
        eval_results = trainer.evaluate(agent, num_episodes=50)
        
        print("\n=== Final Evaluation Results ===")
        print(f"Average Reward: {eval_results['avg_reward']:.3f} ± {eval_results['std_reward']:.3f}")
        print(f"Average Portfolio Value: {eval_results['avg_portfolio_value']:.3f}")
        print(f"Average Sharpe Ratio: {eval_results['avg_sharpe']:.3f}")
        print(f"Success Rate (>100% return): {eval_results['success_rate']:.1%}")
    
    return training_results


if __name__ == "__main__":
    # Example usage
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/data/ticker_hash.pkl"
    
    # Run experiment
    results = run_financial_dqn_experiment(
        data_list_file=data_list_file,
        ticker_hash_file=ticker_hash_file,
        num_episodes=500,
        algorithm='dqn',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )