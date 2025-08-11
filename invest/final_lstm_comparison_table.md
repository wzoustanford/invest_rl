# Complete Model Comparison: Sequential vs LSTM vs DQN/TD3

## Executive Summary

Comparison of investment strategies across different years and configurations:
- **Sequential Supervised**: IIMODEL with 7-day sequences, gamma-discounted Sharpe optimization
- **LSTM Sequential**: LSTM stacked on Conv features, 7-timestep recurrent processing
- **DQN/TD3**: Reinforcement learning approaches

---

## 2021 Performance Comparison

| Model | Configuration | Annual Return | Win Rate | Best Month | Worst Month | Sharpe | Notes |
|-------|--------------|---------------|----------|------------|-------------|--------|-------|
| **Sequential** | γ=0.1 | -42.39% | 42% | +7.85% | -20.65% | - | 12 months |
| **Sequential** | γ=0.3 | -16.22% | 33% | +16.14% | -18.64% | - | 12 months |
| **Sequential** | γ=0.5 | **-7.33%** | 42% | +43.75% | -17.29% | - | 12 months, **best** |
| **LSTM** | γ=0.3 | -21.41%* | 17% | +2.10% | -5.09% | - | 6 months sampled |
| **DQN/TD3** | - | N/A | - | - | - | - | No 2021 data |

*LSTM annual return estimated from 6 sampled months (Feb, Apr, Jun, Aug, Oct, Dec)

### 2021 Key Insights:
- **Best Performer**: Sequential γ=0.5 (-7.33% loss, minimized drawdown in bear market)
- **LSTM Performance**: Underperformed Sequential γ=0.3 by ~5 percentage points
- **Market Context**: High volatility bear market year

---

## 2022 Performance Comparison

| Model | Configuration | Annual Return | Win Rate | Best Month | Worst Month | Sharpe | Notes |
|-------|--------------|---------------|----------|------------|-------------|--------|-------|
| **Sequential** | γ=0.1 | -19.57% | 33% | +13.19% | -18.37% | - | 12 months |
| **Sequential** | γ=0.3 | +14.00% | 42% | +56.98% | -16.19% | - | 12 months |
| **Sequential** | γ=0.5 | **+27.98%** | 60% | +61.59% | -15.20% | - | 10 months, **best** |
| **LSTM** | γ=0.3 | -7.05%* | 50% | +8.55% | -11.68% | - | 6 months sampled |
| **DQN** | - | -8.54% | 0% | - | - | ~0.7 | Apr-Jun window |
| **TD3** | - | -8.54% | 0% | - | - | ~0.7 | Apr-Jun window |

*LSTM annual return estimated from 6 sampled months

### 2022 Key Insights:
- **Best Performer**: Sequential γ=0.5 (+27.98% return)
- **LSTM Performance**: Negative returns while Sequential γ=0.3 achieved +14%
- **RL Methods**: Both DQN and TD3 failed with identical negative returns

---

## 2023 Performance Comparison

| Model | Configuration | Annual Return | Win Rate | Best Month | Worst Month | Sharpe | Notes |
|-------|--------------|---------------|----------|------------|-------------|--------|-------|
| **Sequential** | γ=0.1 | +370.41% | 75% | +117.11% | -6.34% | - | 12 months |
| **Sequential** | γ=0.3 | **+406.17%** | 75% | +81.19% | -9.24% | 0.621† | 12 months, **best** |
| **Sequential** | γ=0.5 | +121.68% | 58% | +99.99% | -28.38% | - | 12 months |
| **Sequential** | Sampled | +35.72%† | 83% | +19.99% | -7.75% | 0.621 | 6 trades |
| **LSTM** | γ=0.3 | TBD | - | - | - | - | Not yet tested |
| **DQN** | - | -0.51% | 0% | - | - | ~0.7 | Mar-May window |
| **TD3** | - | -0.51% | 0% | - | - | ~0.7 | Mar-May window |

†Sampled Sequential from original report (bi-monthly trading)

### 2023 Key Insights:
- **Best Performer**: Sequential γ=0.3 (+406.17% in bull market)
- **RL Methods**: Failed to capture recovery rally

---

## Multi-Year Summary Statistics

### Sequential Supervised (γ=0.3)
- **2021-2023 Average**: +134.65%
- **Best Year**: 2023 (+406.17%)
- **Worst Year**: 2021 (-16.22%)
- **Consistency**: Most balanced across market conditions

### LSTM Sequential (γ=0.3) - Preliminary Results
- **2021-2022 Average**: -14.23% (estimated from samples)
- **Training**: 50 steps (fast mode), needs optimization
- **Architecture**: 64 hidden LSTM, 2 layers
- **Issue**: Underperforming Sequential model significantly

### DQN/TD3
- **Average Return**: -4.52%
- **Win Rate**: 0%
- **Consistency**: Consistently negative across all periods

---

## Model Architecture Comparison

| Model | Parameters | Architecture | Training Time | Objective |
|-------|------------|--------------|---------------|-----------|
| **Sequential** | ~16K | Conv1D → FC → Softmax | Fast (30-750 steps) | Discounted Sharpe |
| **LSTM** | ~50K | Conv1D → LSTM(2) → FC → Softmax | Moderate | Discounted Sharpe |
| **DQN** | ~50K | 3-layer NN | Slow (episodes) | Q-value |
| **TD3** | ~100K | Twin Q-networks + Actor | Slowest | Q-value + Policy |

---

## Recommendations

### Current State Analysis:
1. **Sequential Supervised (γ=0.3)** remains the best performer
2. **LSTM** shows promise but needs optimization:
   - Increase training steps from 50 to 750
   - Tune hyperparameters (hidden dim, layers)
   - Test on full monthly data (not just samples)
3. **RL methods** (DQN/TD3) are unsuitable for this task

### Next Steps for LSTM:
1. Run full 750-iteration training
2. Test all 12 months (not just sampled)
3. Experiment with architecture:
   - Hidden dimensions: 32, 64, 128
   - LSTM layers: 1, 2, 3
   - Dropout rates: 0.0, 0.1, 0.2
4. Test on 2023 bull market data

### Production Recommendation:
- **Continue using Sequential Supervised (γ=0.3)** until LSTM proves superior
- Monthly rebalancing strategy
- Transaction costs: 0.15% per trade
- Risk management: Position sizing and stop-losses

---

## Technical Notes

### LSTM Implementation Details:
- **Feature Extraction**: Exact IIMODEL architecture (Conv → 320 → 47 hidden dims)
- **Architecture Update**: Added hidden layer mapping (320 → 47) before LSTM
- **Temporal Processing**: 7-timestep LSTM sequence with 47-dim input
- **Stock Alignment**: Uses minimum common stocks across timesteps
- **Loss Function**: Sum of gamma-discounted Sharpe ratios
- **Current Limitation**: Only tested with 50 training steps (needs 750)

### Latest LSTM Results (with updated architecture):
- **2021**: -26.43% (50 steps, 6 months sampled)
- **2022**: 0.04% (50 steps, 6 months sampled)
- **Improvement**: 2022 went from -7% to 0% with architecture alignment

### Data Consistency:
- All models tested on same S&P 500 universe
- 25-day holding period
- 360-day training window
- Transaction costs: 0.15%

---

*Report Generated: August 10, 2025*  
*LSTM Results: Preliminary (50 training steps, sampled months only)*  
*Full LSTM evaluation pending with 750 training steps*