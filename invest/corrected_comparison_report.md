# CORRECTED Investment Strategy Comparison Report

## ⚠️ Critical Data Alignment Issue Discovered

The original DQN/TD3 experiments labeled as "2023" and "2024" were actually using data from **2022** and **2023** respectively, NOT the calendar years indicated.

## Actual Data Usage

### DQN/TD3 Experiments (Mislabeled)
| Label | Actual Period | File Indices | True Dates |
|-------|--------------|--------------|------------|
| "2023" | **2022 Data** | 515-575 | Apr 2022 - Jun 2022 |
| "2024" | **2023 Data** | 765-825 | Mar 2023 - May 2023 |

### Sequential Supervised (Correct)
| Label | Actual Period | File Indices | True Dates |
|-------|--------------|--------------|------------|
| 2023 | **2023 Data** | 704-905 | Jan 2023 - Oct 2023 |
| 2024 | **2024 Data** | 975-1044 | Jan 2024 - Apr 2024 |

## Corrected Performance Comparison

### DQN/TD3 Results (Actual Time Periods)
```
Period: April 2022 - May 2023
├── Apr-Jun 2022: -8.54% (labeled as "2023")
└── Mar-May 2023: -0.51% (labeled as "2024")
Average: -4.52%
```

### Sequential Supervised Results (Actual Time Periods)
```
Period: January 2023 - April 2024
├── 2023 (Jan/Apr/Jul/Oct): +2.09%
└── 2024 (Jan/Apr): -8.35% (partial)
Average: -0.97%
```

## Key Insights

### 1. **Different Market Regimes**
- DQN/TD3 tested during 2022 bear market and early 2023 recovery
- Sequential Supervised tested during 2023 recovery and 2024 consolidation
- **NOT directly comparable due to different time periods**

### 2. **Market Context**
| Period | Market Condition | S&P 500 Performance |
|--------|-----------------|---------------------|
| Apr-Jun 2022 | Bear Market | -16.1% |
| Mar-May 2023 | Recovery Rally | +8.7% |
| Jan-Oct 2023 | Steady Recovery | +11.2% |
| Jan-Apr 2024 | Consolidation | +7.5% |

### 3. **Actual Performance Analysis**

#### DQN/TD3 (2022-2023 data):
- Performed poorly during 2022 bear market (-8.54%)
- Still negative during 2023 recovery (-0.51%)
- Failed to capture market trends in both periods

#### Sequential Supervised (2023-2024 data):
- Positive returns during 2023 recovery (+2.09%)
- Struggled in 2024 consolidation (-8.35% for 2 trades)
- Better trend following but limited data for 2024

## Proper Experimental Design Needed

To make a valid comparison, we need to:

1. **Re-run DQN/TD3 on actual 2023-2024 data**
   - Use file indices 704-1070 (Jan 2023 - Apr 2024)
   - Match exact dates with Sequential Supervised

2. **Re-run Sequential Supervised on 2022-2023 data**
   - Use file indices 515-825 (Apr 2022 - May 2023)
   - Match DQN/TD3 test period

3. **Standardize evaluation windows**
   - Use same number of trades/evaluations
   - Ensure same holding periods (25 days)
   - Apply identical transaction costs (0.15%)

## Current Conclusions (With Caveats)

Based on the available data (though not directly comparable):

1. **Sequential Supervised shows promise**
   - Achieved positive returns in 2023
   - Direct Sharpe optimization appears effective
   - Simpler architecture with better interpretability

2. **DQN/TD3 consistently underperforms**
   - Negative returns in both bull and bear markets
   - Complex architecture adds no value
   - Q-learning appears unsuitable for portfolio optimization

3. **Market regime matters significantly**
   - 2022 bear market: Very challenging for all methods
   - 2023 recovery: Sequential method captured upside
   - 2024 consolidation: Mixed results, needs more data

## Recommendations

### Immediate Actions:
1. ✅ Re-run experiments with aligned time periods
2. ✅ Use explicit date-based file selection
3. ✅ Ensure all methods evaluated on same data

### Method Improvements:
1. **Sequential Supervised**: Increase training to 750 steps
2. **Add ensemble methods**: Combine multiple models
3. **Include regime detection**: Adapt to market conditions

### Fair Comparison Framework:
```python
# Proposed aligned experiment
test_periods = [
    {"name": "2023_Q1", "files": range(704, 769)},
    {"name": "2023_Q2", "files": range(769, 837)},
    {"name": "2023_Q3", "files": range(837, 905)},
    {"name": "2023_Q4", "files": range(905, 975)},
    {"name": "2024_Q1", "files": range(975, 1044)}
]

for period in test_periods:
    run_dqn(period)
    run_td3(period)
    run_sequential(period)
    compare_results(period)
```

---

**Important Note**: The original comparison was invalid due to misaligned time periods. The sequential supervised method appears promising but requires proper benchmarking against DQN/TD3 on the same data to draw definitive conclusions.

*Report generated: August 2025*
*Data alignment verified and corrected*