"""
Simple unit test for the financial environment.
Tests basic functionality and verifies the environment interface.
"""

import torch
import numpy as np
import pickle
import os
import sys
from typing import Dict, List

# Import the financial environment
from financial_env import FinancialTradingEnvironment, FinancialEnvironmentWrapper, create_financial_environment


def create_dummy_data_files(num_days: int = 10) -> tuple:
    """Create dummy data files for testing."""
    # Create test directory
    test_dir = "/tmp/financial_env_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create data list file
    data_list_file = os.path.join(test_dir, "test_data_list.txt")
    data_files = []
    
    with open(data_list_file, 'w') as f:
        for i in range(num_days):
            data_file = os.path.join(test_dir, f"day_{i}.pkl")
            data_files.append(data_file)
            f.write(data_file + '\n')
            
            # Create dummy data for each day
            num_stocks = 5
            feature_dim = 249
            time_steps = 10
            
            dummy_data = {
                'trainFeature': torch.randn(num_stocks, feature_dim),
                'train_in_portfolio_series': torch.abs(torch.randn(num_stocks, time_steps)) * 100 + 50,  # Prices
                'all_train_tickers': [f'STOCK_{j}' for j in range(num_stocks)]
            }
            
            with open(data_file, 'wb') as df:
                pickle.dump(dummy_data, df)
    
    # Create ticker hash file
    ticker_hash_file = os.path.join(test_dir, "test_ticker_hash.pkl")
    ticker_hash = {
        'hash_D': {f'STOCK_{i}': i for i in range(10)},  # Support up to 10 stocks
        'num_tickers': 10
    }
    
    with open(ticker_hash_file, 'wb') as f:
        pickle.dump(ticker_hash, f)
    
    return data_list_file, ticker_hash_file, test_dir


def test_environment_creation():
    """Test basic environment creation."""
    print("=== Test 1: Environment Creation ===")
    
    data_list_file, ticker_hash_file, test_dir = create_dummy_data_files()
    
    try:
        # Create environment
        env = FinancialTradingEnvironment(
            data_list_filename=data_list_file,
            ticker_hash_file=ticker_hash_file,
            start_date_idx=0,
            end_date_idx_plus1=5,
            device='cpu'
        )
        
        print("✓ Environment created successfully")
        print(f"  Number of tickers: {env.num_tickers}")
        print(f"  Data range: {env.start_date_idx} to {env.end_date_idx_plus1}")
        
        # Cleanup
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_reset_and_step():
    """Test environment reset and step functionality."""
    print("\n=== Test 2: Reset and Step ===")
    
    data_list_file, ticker_hash_file, test_dir = create_dummy_data_files()
    
    try:
        # Create environment
        env = FinancialTradingEnvironment(
            data_list_filename=data_list_file,
            ticker_hash_file=ticker_hash_file,
            start_date_idx=0,
            end_date_idx_plus1=5,
            device='cpu'
        )
        
        # Test reset (dict mode)
        state, info = env.reset(return_dict=True)
        print("✓ Reset (dict mode) successful")
        print(f"  State keys: {list(state.keys())}")
        print(f"  Initial X: {state['X']}")
        
        # Test step with random action
        action = torch.softmax(torch.randn(env.num_tickers), dim=0)  # Random portfolio
        next_state, reward, terminated, truncated, step_info = env.step(action, return_dict=True)
        
        print("✓ Step (dict mode) successful")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio value X: {step_info['X']:.4f}")
        print(f"  Sharpe: {step_info['sharpe']:.4f}")
        
        # Test reset (flat mode)
        flat_obs, info = env.reset(return_dict=False)
        print("\n✓ Reset (flat mode) successful")
        print(f"  Observation shape: {flat_obs.shape}")
        print(f"  Expected dim: {env.get_observation_dim()}")
        
        # Test step with flat observation
        next_obs, reward, terminated, truncated, step_info = env.step(action, return_dict=False)
        print("✓ Step (flat mode) successful")
        print(f"  Next observation shape: {next_obs.shape}")
        
        # Cleanup
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Reset/Step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_wrapper_dqn_mode():
    """Test environment wrapper in DQN mode."""
    print("\n=== Test 3: Wrapper DQN Mode ===")
    
    data_list_file, ticker_hash_file, test_dir = create_dummy_data_files()
    
    try:
        # Create wrapper in DQN mode
        env = FinancialEnvironmentWrapper(
            data_list_filename=data_list_file,
            ticker_hash_file=ticker_hash_file,
            use_flat_obs=True,
            discrete_actions=100,
            start_date_idx=0,
            end_date_idx_plus1=5,
            device='cpu'
        )
        
        print("✓ Wrapper created in DQN mode")
        print(f"  Action space: {env.action_space.n} discrete actions")
        print(f"  Observation shape: {env.observation_space.shape}")
        
        # Test reset
        obs, info = env.reset()
        print("✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        
        # Test step with discrete action
        discrete_action = np.random.randint(0, env.action_space.n)
        next_obs, reward, terminated, truncated, info = env.step(discrete_action)
        
        print("✓ Step with discrete action successful")
        print(f"  Discrete action: {discrete_action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Next observation shape: {next_obs.shape}")
        
        # Run a few more steps
        total_reward = reward
        for _ in range(3):
            action = np.random.randint(0, env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break
        
        print(f"✓ Multiple steps completed")
        print(f"  Total reward over 4 steps: {total_reward:.4f}")
        
        # Cleanup
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Wrapper DQN mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_wrapper_original_mode():
    """Test environment wrapper in original mode."""
    print("\n=== Test 4: Wrapper Original Mode ===")
    
    data_list_file, ticker_hash_file, test_dir = create_dummy_data_files()
    
    try:
        # Create wrapper in original mode
        env = FinancialEnvironmentWrapper(
            data_list_filename=data_list_file,
            ticker_hash_file=ticker_hash_file,
            use_flat_obs=False,
            start_date_idx=0,
            end_date_idx_plus1=5,
            device='cpu'
        )
        
        print("✓ Wrapper created in original mode")
        print(f"  Action space: continuous ({env.num_tickers} stocks)")
        print(f"  Observation keys: {env.observation_space.keys}")
        
        # Test reset
        state, info = env.reset()
        print("✓ Reset successful")
        print(f"  State type: {type(state)}")
        print(f"  State keys: {list(state.keys())}")
        
        # Test step with continuous action
        action = torch.softmax(torch.randn(env.num_tickers), dim=0)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print("✓ Step with continuous action successful")
        print(f"  Action shape: {action.shape}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Next state keys: {list(next_state.keys())}")
        
        # Cleanup
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Wrapper original mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_replay_transition():
    """Test replay buffer transition creation."""
    print("\n=== Test 5: Replay Transition ===")
    
    data_list_file, ticker_hash_file, test_dir = create_dummy_data_files()
    
    try:
        # Create environment
        env = FinancialTradingEnvironment(
            data_list_filename=data_list_file,
            ticker_hash_file=ticker_hash_file,
            start_date_idx=0,
            end_date_idx_plus1=5,
            device='cpu'
        )
        
        # Reset
        state, info = env.reset()
        
        # First step (no transition yet)
        action = torch.softmax(torch.randn(env.num_tickers), dim=0)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        transition = env.get_transition_for_replay()
        if transition is None:
            print("✓ No transition on first step (expected)")
        else:
            print("✗ Unexpected transition on first step")
        
        # Second step (should have transition)
        action = torch.softmax(torch.randn(env.num_tickers), dim=0)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        transition = env.get_transition_for_replay()
        if transition is not None:
            print("✓ Transition created after second step")
            print(f"  Transition keys: {list(transition.keys())}")
            print(f"  Prev state X: {transition['prev_state']['X']}")
            print(f"  Current state X: {transition['state']['X']}")
        else:
            print("✗ No transition after second step")
        
        # Cleanup
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Replay transition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Financial Environment Unit Tests")
    print("=" * 50)
    
    tests = [
        test_environment_creation,
        test_reset_and_step,
        test_wrapper_dqn_mode,
        test_wrapper_original_mode,
        test_replay_transition
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
        print("🎉 All tests passed! Financial environment is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)