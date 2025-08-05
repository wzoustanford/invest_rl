"""
Test script to verify financial DQN integration works correctly.
"""

import sys
import os
import torch
import numpy as np

# Add current directory to path
sys.path.append('/home/ubuntu/code/angle_rl/invest')

from dqn_financial_integration import FinancialDQNTrainer, run_financial_dqn_experiment
from model.dqn_financial_wrapper import FinancialDQNWrapper
from model.financial_policy_model import create_financial_policy_model


def test_environment_creation():
    """Test if financial environment can be created"""
    print("=== Testing Environment Creation ===")
    
    # Check if data files exist
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/data/ticker_hash.pkl" 
    
    if not os.path.exists(data_list_file):
        print(f"WARNING: Data list file not found: {data_list_file}")
        print("Creating dummy data list file for testing...")
        
        # Create dummy data list
        os.makedirs(os.path.dirname(data_list_file), exist_ok=True)
        with open(data_list_file, 'w') as f:
            f.write("/dummy/path/data1.pkl\n")
            f.write("/dummy/path/data2.pkl\n")
    
    if not os.path.exists(ticker_hash_file):
        print(f"WARNING: Ticker hash file not found: {ticker_hash_file}")
        print("Creating dummy ticker hash file for testing...")
        
        import pickle
        dummy_hash = {
            'hash_D': {'AAPL': 0, 'GOOGL': 1, 'MSFT': 2, 'TSLA': 3, 'AMZN': 4},
            'num_tickers': 5
        }
        os.makedirs(os.path.dirname(ticker_hash_file), exist_ok=True)
        with open(ticker_hash_file, 'wb') as f:
            pickle.dump(dummy_hash, f)
    
    try:
        # Test environment creation
        env = FinancialDQNWrapper(
            data_list_file=data_list_file,
            ticker_hash_file=ticker_hash_file,
            feature_aggregation='mean',
            max_episode_steps=10,
            device='cpu'
        )
        
        print(f"✓ Environment created successfully")
        print(f"  Action space: {env.action_space.n}")
        print(f"  Observation space: {env.observation_space}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False


def test_policy_model_creation():
    """Test if financial policy model can be created"""
    print("\n=== Testing Policy Model Creation ===")
    
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/data/ticker_hash.pkl"
    
    try:
        policy_model = create_financial_policy_model(
            data_list_file=data_list_file,
            ticker_hash_file=ticker_hash_file,
            device='cpu'
        )
        
        print(f"✓ Policy model created successfully")
        print(f"  Input dim: {policy_model.obs_dim}")
        print(f"  Action space: {policy_model.n_actions}")
        print(f"  Device: {policy_model.device}")
        
        # Test forward pass (set to eval mode to avoid BatchNorm issues)
        policy_model.set_training_mode(False)
        dummy_obs = torch.randn(policy_model.obs_dim)
        action = policy_model.select_action(dummy_obs, epsilon=0.1)
        
        print(f"✓ Action selection works: {action}")
        
        return True
        
    except Exception as e:
        print(f"✗ Policy model creation failed: {e}")
        return False


def test_trainer_creation():
    """Test if DQN trainer can be created"""
    print("\n=== Testing Trainer Creation ===")
    
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/data/ticker_hash.pkl"
    
    try:
        trainer = FinancialDQNTrainer(
            data_list_file=data_list_file,
            ticker_hash_file=ticker_hash_file,
            algorithm='dqn',
            device='cpu'
        )
        
        print(f"✓ Trainer created successfully")
        print(f"  Algorithm: {trainer.algorithm}")
        print(f"  Device: {trainer.devmgr.device}")
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        return False


def test_angle_rl_imports():
    """Test if we can import from angle/RL repository"""
    print("\n=== Testing angle/RL Imports ===")
    
    try:
        # Test device utils import
        sys.path.append('/home/ubuntu/code/angle/RL')
        from model.device_utils import get_device_manager
        
        devmgr = get_device_manager('cpu')
        print(f"✓ Device manager imported: {devmgr.device}")
        
        # Test config import
        from config.AgentConfig import AgentConfig
        config = AgentConfig()
        print(f"✓ AgentConfig imported: gamma={config.gamma}")
        
        return True
        
    except Exception as e:
        print(f"✗ angle/RL imports failed: {e}")
        return False


def test_basic_integration():
    """Test basic integration without actual training"""
    print("\n=== Testing Basic Integration ===")
    
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/data/ticker_hash.pkl"
    
    try:
        trainer = FinancialDQNTrainer(
            data_list_file=data_list_file,
            ticker_hash_file=ticker_hash_file,
            algorithm='dqn',
            device='cpu'
        )
        
        # Test environment reset
        obs, info = trainer.env.reset()
        print(f"✓ Environment reset works, obs shape: {obs.shape}")
        
        # Test action selection (set to eval mode)
        trainer.policy_model.set_training_mode(False)
        action = trainer.policy_model.select_action(torch.from_numpy(obs))
        print(f"✓ Action selection works: {action}")
        
        # Test portfolio decoding
        portfolio = trainer.policy_model.decode_action(action)
        print(f"✓ Portfolio decoding works, sum: {portfolio.sum():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic integration test failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests"""
    print("Starting Financial DQN Integration Tests...")
    print("=" * 50)
    
    tests = [
        test_angle_rl_imports,
        test_environment_creation,
        test_policy_model_creation,
        test_trainer_creation,
        test_basic_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__:<30} [{status}]")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Financial DQN integration is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)