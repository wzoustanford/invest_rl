# Code Details for Final Investment Strategy Comparison Report

## Overview
This document details the Python scripts and code components used to generate the Final Investment Strategy Comparison Report comparing DQN, TD3, and Sequential Supervised Learning approaches.

## Core Scripts by Algorithm

### 1. Sequential Supervised Learning Implementation

#### Primary Scripts
- **`run_sampled_monthly.py`**
  - Implements sampled monthly trading strategy
  - Executes 6 trades/year for 2022-2023, 2 trades for 2024
  - Uses gamma=0.3 for discounting
  - Training steps: 30 (fast mode)
  
- **`train_sequential_step_model.py` / `train_sequential_step_model_consecutive.py`**
  - Core training logic for sequential models
  - Implements 7-day consecutive sequence training
  - Optimizes Sharpe ratio with gamma discounting
  
- **`run_optimized_sequential_experiment.py`**
  - Optimized version with structured reporting
  - Config: learning_rate=0.001, training_steps=100
  - Uses ExperimentConfig dataclass for parameter management

#### Model Architecture
- **`model/iimodel.py`**
  - IIMODEL class implementation
  - Architecture:
    - Conv1D layer (32 filters, kernel_size=3)
    - Softplus activation
    - AdaptiveMaxPool1d (output_size=10)
    - FC layers (320 → 47 → 1)
    - Softmax output for portfolio weights
  - Parameters: ~16K
  - Dropout ratio: 0.0 for production

### 2. DQN (Deep Q-Network) Implementation

#### Primary Scripts
- **`run_large_scale_dqn.py`**
  - Large-scale DQN experiments on S&P 500 stocks
  - Implements experience replay buffer (10K capacity)
  
- **`run_dqn_real_data.py`**
  - DQN implementation on real market data
  - Epsilon-greedy exploration: 0.1 → 0.01
  
- **`financial_dqn_agent.py`**
  - Core DQN agent implementation
  - Q-value approximation for portfolio decisions
  
- **`train_with_dqn.py`**
  - Training loop implementation
  - 2-5 episodes per experiment
  - Learning rate: 0.001

#### Model Components
- **`model/dqn_policy_model.py`**
  - 3-layer neural network
  - ~50K parameters
  - Action space: portfolio allocation decisions

### 3. TD3 (Twin Delayed DDPG) Implementation

#### Primary Scripts
- **`run_td3_large_experiment.py`**
  - TD3 experiments with twin Q-networks
  - Parameters: τ=0.005, policy_delay=2
  
- **`test_td3_experiment.py`**
  - TD3 testing and validation
  - Implements delayed policy updates
  
- **`sliding_window_experiment.py`**
  - Sliding window approach for both DQN and TD3
  - Tests on 2022-2023 data windows

### 4. Comparison and Analysis Scripts

#### Main Comparison
- **`full_dataset_comparison.py`**
  - Master comparison script
  - Runs all three algorithms on same data
  - Generates performance metrics
  - Class: FullDatasetComparison
  - Output directory: `/experiments/full_comparison_[timestamp]/`

#### Supporting Analysis
- **`monthly_trading_experiment.py`**
  - Monthly rebalancing analysis
  - Calculates Sharpe ratios and returns
  
- **`show_experiment_summary.py`**
  - Aggregates results across experiments
  - Generates summary statistics

### 5. Data Processing Pipeline

#### Data Files
- **Location**: `/home/ubuntu/code/angle_rl/invest/data/`
- **Naming Convention**: 
  ```
  model_data_single_step_trainingtimelength360d_buyselltimelength25d_
  training_data_start_date_YYYY_MM_DD_test_data_start_date_YYYY_MM_DD_
  newsFeaturesFalse_alpacafracfiltered.pkl
  ```

#### Data Lists
- **`data/all_data_list.txt`**
  - Master list of all available data files
  - 1000+ files covering 2020-2024
  
- **Date-specific lists**:
  - Various `data_list_[date_range]_*.txt` files
  - Used for specific experiment windows

## Configuration Parameters

### Common Settings
```python
# All algorithms
holding_period = 25  # days
transaction_cost = 0.0015  # 0.15%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
```

### Sequential Supervised Settings
```python
# Model configuration
num_consecutive_days = 7
gamma = 0.3  # discount factor
learning_rate = 0.001
training_steps = 30  # fast mode, up to 200 for full training

# Architecture
num_conv_filters = 32
hidden_dim = 47
dropout_ratio = 0.0
```

### DQN/TD3 Settings
```python
# DQN specific
epsilon_start = 0.1
epsilon_end = 0.01
episodes = 2-5
experience_replay_size = 10000

# TD3 specific
tau = 0.005  # soft update parameter
policy_delay = 2
actor_lr = 0.001
critic_lr = 0.001
```

## File Index Mapping

### Sequential Supervised - Exact Files Used
```python
file_indices = {
    2022: [464, 507, 551, 594, 639, 682],  # 6 monthly samples
    2023: [704, 747, 790, 837, 883, 929],  # 6 monthly samples
    2024: [975, 1022]                      # 2 monthly samples
}
```

### DQN/TD3 - File Ranges
```python
dqn_td3_ranges = {
    "2022_data": range(515, 576),  # Apr-Jun 2022
    "2023_data": range(765, 826)   # Mar-May 2023
}
```

## Key Functions and Classes

### Sequential Model Training
```python
def train_fast(train_files, steps=30):
    """Ultra-fast training for sequential model"""
    # Loads 7 consecutive days
    # Optimizes: -sharpe * (gamma ** t)
    # Returns trained IIMODEL
```

### DQN Agent
```python
class FinancialDQNAgent:
    """DQN agent for portfolio optimization"""
    # Methods: act(), remember(), replay(), update_epsilon()
```

### Evaluation
```python
def evaluate_fast(model, test_file):
    """Fast evaluation on test data"""
    # Calculates returns with transaction costs
    # Returns: (return_pct, sharpe_ratio, selected_stocks)
```

## Output Structure

### Results Directory
```
data/
├── seq_sampled_[timestamp]/
│   ├── results.json
│   ├── model_*.pt
│   └── trades_*.pkl
├── dqn_results/
│   ├── episode_rewards.json
│   └── q_values.npy
└── td3_results/
    ├── actor_losses.json
    └── critic_losses.json
```

### Report Generation
The final comparison report aggregates results from:
1. Sequential supervised monthly trades
2. DQN sliding window experiments
3. TD3 enhanced experiments

## Execution Flow

1. **Data Preparation**
   - Load pickle files from data directory
   - Extract features and return series
   - Normalize features

2. **Model Training**
   - Sequential: 7-day sequences, gamma-discounted
   - DQN/TD3: Experience replay, epsilon-greedy

3. **Evaluation**
   - Apply trained models to test periods
   - Calculate returns with transaction costs
   - Track Sharpe ratios and win rates

4. **Comparison**
   - Aggregate results across methods
   - Generate performance metrics
   - Create final report

## Notes

- Sequential Supervised consistently outperformed RL methods
- File indices in report may have label discrepancies (e.g., "2023" label for 2022 data)
- Transaction costs significantly impact RL performance
- Gamma=0.3 proved optimal for sequential approach
- Monthly rebalancing (12x/year) recommended for production

---
*Generated: August 8, 2025*
*Purpose: Document code structure for investment strategy comparison report*