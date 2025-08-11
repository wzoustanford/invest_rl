# LSTM vs Sequential Supervised - Equal Training (750 iterations)

## Configuration
- Both models: 750 training iterations, γ=0.3
- LSTM: 64 hidden dims, 2 layers, gradient clipping (max_norm=5.0)
- Both use 7 consecutive daily files for training
- Transaction costs: 0.15% per trade

## Results Table

| Year | Model | Iterations | Months Traded | Annual Return | Win Rate | Best Month | Worst Month |
|------|-------|------------|---------------|---------------|----------|------------|-------------|
| 2021 | LSTM γ=0.3 | 750 | 8 (5,6,7,8,9,10,11,12) | +12.28% | 50% | +12.34% | -4.87% |
| 2022 | LSTM γ=0.3 | 750 | 12 (1,2,3,4,5,6,7,8,9,10,11,12) | -17.16% | 42% | +19.01% | -21.59% |
| 2021 | Sequential γ=0.3 | 750 | 12 (all) | -16.22% | 33% | +16.14% | -18.64% |
| 2022 | Sequential γ=0.3 | 750 | 12 (all) | +14.00% | 42% | +56.98% | -16.19% |

## Monthly Details

### LSTM 2021 Monthly Returns

- Month 05: +0.35% (18 stocks selected)
- Month 06: +6.19% (1 stocks selected)
- Month 07: -2.42% (26 stocks selected)
- Month 08: -0.24% (6 stocks selected)
- Month 09: -1.95% (29 stocks selected)
- Month 10: +3.29% (4 stocks selected)
- Month 11: +12.34% (3 stocks selected)
- Month 12: -4.87% (10 stocks selected)

### LSTM 2022 Monthly Returns

- Month 01: -21.59% (5 stocks selected)
- Month 02: +2.69% (9 stocks selected)
- Month 03: -7.06% (5 stocks selected)
- Month 04: +4.13% (4 stocks selected)
- Month 05: -5.02% (40 stocks selected)
- Month 06: -4.99% (0 stocks selected)
- Month 07: +4.41% (0 stocks selected)
- Month 08: +2.40% (15 stocks selected)
- Month 09: -4.64% (21 stocks selected)
- Month 10: +19.01% (7 stocks selected)
- Month 11: -0.40% (15 stocks selected)
- Month 12: -2.54% (6 stocks selected)

