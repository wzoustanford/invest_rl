"""
Test script to verify DQN integration with financial environment.
This tests that angle/RL's DQN can work with angle_rl's financial data.
"""

import sys
import os
import torch
import numpy as np

# Add paths
sys.path.append('/home/ubuntu/code/angle/RL')
sys.path.append('/home/ubuntu/code/angle_rl/invest')

# Import test utilities
from test_financial_env import create_dummy_data_files


def test_dqn_import():
    """Test if we can import DQN components."""
    print("=== Test 1: DQN Import ===")
    
    try:
        # Test our financial DQN and components
        from financial_dqn_agent import create_financial_dqn_agent, DQNNetwork
        print("✓ Successfully imported financial DQN components")
        
        # Test we can create a network
        network = DQNNetwork(input_dim=100, output_dim=10, use_dueling=True)
        print("✓ Successfully created DQN network")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import DQN components: {e}")
        return False


def test_financial_adapter():
    """Test if financial DQN agent works."""
    print("\n=== Test 2: Financial DQN Agent ===")
    
    try:
        from financial_dqn_agent import FinancialDQNAgent
        
        # Create agent
        agent = FinancialDQNAgent(
            observation_dim=100,
            n_actions=50,
            memory_size=1000,
            device='cpu'
        )
        print("✓ Created FinancialDQNAgent")
        
        # Test action selection
        obs = np.random.randn(100).astype(np.float32)
        action = agent.select_action(obs)
        print(f"✓ Action selection works: {action}")
        
        # Test experience storage
        next_obs = np.random.randn(100).astype(np.float32)
        agent.store_experience(obs, action, 0.5, next_obs, False)
        print(f"✓ Experience stored, memory size: {len(agent.memory)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Financial DQN agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dqn_with_financial_env():
    """Test DQN agent with financial environment."""
    print("\n=== Test 3: DQN with Financial Environment ===")
    
    data_list_file, ticker_hash_file, test_dir = create_dummy_data_files(num_days=20)
    
    try:
        from financial_env import create_financial_environment
        from financial_dqn_agent import create_financial_dqn_agent
        
        # Create environment
        env = create_financial_environment(
            data_list_filename=data_list_file,
            ticker_hash_file=ticker_hash_file,
            use_flat_obs=True,
            discrete_actions=50,
            start_date_idx=0,
            end_date_idx_plus1=10,
            device='cpu'
        )
        
        print("✓ Created financial environment for DQN")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.n}")
        
        # Create DQN agent
        agent = create_financial_dqn_agent(
            observation_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            gamma=0.8,
            epsilon_start=1.0,
            epsilon_end=0.1,
            lr=0.001,
            batch_size=16,
            memory_size=1000,
            device='cpu'
        )
        print("✓ Created DQN agent")
        
        # Test episode
        obs, info = env.reset()
        total_reward = 0.0
        
        for step in range(5):
            # Select action
            action = agent.select_action(obs)
            print(f"  Step {step}: Selected action {action}")
            
            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        print(f"✓ Completed test episode")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Memory size: {len(agent.memory)}")
        
        # Test training
        if len(agent.memory) >= agent.batch_size:
            loss = agent.train()
            print(f"✓ Training step completed, loss: {loss:.4f}")
        
        # Cleanup
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ DQN integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_training_script():
    """Test the full training script."""
    print("\n=== Test 4: Training Script ===")
    
    data_list_file, ticker_hash_file, test_dir = create_dummy_data_files(num_days=20)
    
    try:
        from train_with_dqn import train_financial_dqn, evaluate_financial_dqn
        
        # Run short training
        agent, history = train_financial_dqn(
            data_list_filename=data_list_file,
            ticker_hash_file=ticker_hash_file,
            exp_id="test_exp",
            start_date_idx=0,
            end_date_idx_plus1=10,
            eval_start_date_idx=10,
            eval_end_date_idx_plus1=15,
            num_episodes=2,  # Very short for testing
            num_discrete_actions=50,
            batch_size=8,
            memory_size=100,
            log_interval=1,
            save_interval=10,
            device='cpu'
        )
        
        print("✓ Training completed")
        print(f"  Episodes run: {len(history['episode_rewards'])}")
        print(f"  Final reward: {history['episode_rewards'][-1]:.4f}")
        
        # Test evaluation
        eval_results = evaluate_financial_dqn(
            agent=agent,
            data_list_filename=data_list_file,
            ticker_hash_file=ticker_hash_file,
            eval_start_date_idx=10,
            eval_end_date_idx_plus1=15,
            num_discrete_actions=50,
            device='cpu'
        )
        
        print("✓ Evaluation completed")
        print(f"  Final portfolio value: {eval_results['final_portfolio_value']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        # Cleanup test output
        test_dir = "/home/ubuntu/code/angle_rl/invest/data/test_exp"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def run_all_tests():
    """Run all integration tests."""
    print("=" * 50)
    print("DQN Financial Integration Tests")
    print("=" * 50)
    
    tests = [
        test_dqn_import,
        test_financial_adapter,
        test_dqn_with_financial_env,
        test_training_script
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
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__:<30} [{status}]")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DQN integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)