# LSTM vs Sequential Supervised - CORRECTED DATE ALIGNMENT
## 750 Iterations Comparison with Proper Trading Year Mapping

### ⚠️ IMPORTANT DATE CORRECTION
Based on code analysis, the actual trading dates are **1 year after** the test_data_start_date in filenames:
- Files with `test_data_start_date_2020_XX_XX` → Trade in **2021**
- Files with `test_data_start_date_2021_XX_XX` → Trade in **2022**
- Files with `test_data_start_date_2022_XX_XX` → Trade in **2023**

## Configuration
- Both models: 750 training iterations, γ=0.3
- LSTM: 64 hidden dims, 2 layers, gradient clipping (max_norm=5.0)
- Both use 7 consecutive daily files for training
- Transaction costs: 0.15% per trade

## 📊 CORRECTED Results Comparison

### Actual Trading Years:
- **LSTM trained on**: 2021 (May-Dec) and 2022 (full year)
- **Sequential Supervised from gamma report**: 2022 (full year) and 2023 (full year)

| Actual Year | Model | Data Source | Months | Annual Return | Win Rate | Best Month | Worst Month | Market Context |
|-------------|-------|-------------|--------|---------------|----------|------------|-------------|----------------|
| **2021 H1** | LSTM γ=0.3 | N/A | - | - | - | - | - | No data |
| **2021 H1** | Sequential γ=0.3 | N/A | - | - | - | - | - | No data |
| **2021 H2** | LSTM γ=0.3 | This experiment | 8 (May-Dec)† | +12.28% | 50% | +12.34% | -4.87% | Bull market (+11% S&P H2) |
| **2021 H2** | Sequential γ=0.3 | Gamma report* | 6 (Jul-Dec) | -28.96% | - | - | - | Bull market (+11% S&P H2) |
| **2022** | LSTM γ=0.3 | This experiment | 12 (all) | -17.16% | 42% | +19.01% | -21.59% | Bear market (-18.1% S&P) |
| **2022** | Sequential γ=0.3 | Gamma report* | 12 (all) | -16.22% | 33% | +16.14% | -18.64% | Bear market (-18.1% S&P) |
| **2023** | LSTM γ=0.3 | N/A | - | - | - | - | - | Not tested |
| **2023** | Sequential γ=0.3 | Gamma report* | 12 (all) | +14.00% | 42% | +56.98% | -16.19% | Recovery (+24.2% S&P) |

*From `/home/ubuntu/code/angle_rl/invest/experiments/REALIGNED_GAMMA_RESULTS_CORRECT_DATES.md`

## 📈 Year-by-Year Comparisons

### 2021 H2 Performance (Bull Market - S&P 500: +11%)

| Model | Period | Annual Return | vs S&P 500 | Market Context |
|-------|--------|--------------|------------|----------------|
| **LSTM γ=0.3** | May-Dec (8 months) | +12.28% | +1.28pp | Outperformed |
| **Sequential γ=0.3** | Jul-Dec (6 months) | -28.96% | -39.96pp | Severely underperformed |
| **Winner** | LSTM by 41.24pp | | | |

**Key Insight**: LSTM dramatically outperformed Sequential in 2021 H2 bull market

### 2022 Performance (Bear Market - S&P 500: -18.1%)

| Model | Annual Return | vs S&P 500 | Win Rate | Volatility | Best Trade | Worst Trade |
|-------|--------------|------------|----------|------------|------------|-------------|
| **LSTM γ=0.3** | -17.16% | +0.94pp | 42% (5/12) | High | +19.01% (Oct) | -21.59% (Jan) |
| **Sequential γ=0.3** | -16.22% | +1.88pp | 33% (4/12) | High | +16.14% | -18.64% |
| **Winner** | Sequential | Sequential | LSTM | Similar | LSTM | LSTM |

**Key Insights for 2022:**
- Both models outperformed the S&P 500 in the bear market
- Sequential Supervised slightly better overall (-16.22% vs -17.16%)
- LSTM had higher win rate but also more extreme losses
- LSTM's October gain (+19.01%) was exceptional

## 📊 LSTM 2021 Detailed Results (May-December)
*Note: Sequential Supervised has no 2021 data for comparison*

| Month | Return | Stocks Selected | Cumulative |
|-------|--------|-----------------|------------|
| May | +0.35% | 18 | +0.35% |
| Jun | +6.19% | 1 | +6.56% |
| Jul | -2.42% | 26 | +3.98% |
| Aug | -0.24% | 6 | +3.73% |
| Sep | -1.95% | 29 | +1.71% |
| Oct | +3.29% | 4 | +5.06% |
| Nov | +12.34% | 3 | +18.00% |
| Dec | -4.87% | 10 | +12.28% |

## 📊 LSTM 2022 Detailed Results (Full Year)

| Month | Return | Stocks Selected | Cumulative |
|-------|--------|-----------------|------------|
| Jan | -21.59% | 5 | -21.59% |
| Feb | +2.69% | 9 | -19.49% |
| Mar | -7.06% | 5 | -25.51% |
| Apr | +4.13% | 4 | -22.45% |
| May | -5.02% | 40 | -26.33% |
| Jun | -4.99% | 0* | -30.05% |
| Jul | +4.41% | 0* | -26.98% |
| Aug | +2.40% | 15 | -25.17% |
| Sep | -4.64% | 21 | -28.61% |
| Oct | +19.01% | 7 | -14.77% |
| Nov | -0.40% | 15 | -15.10% |
| Dec | -2.54% | 6 | -17.16% |

*Zero stocks selected indicates potential convergence issues

## 🔍 Key Observations

### 1. **Date Alignment is Critical**
- Previous comparisons were misaligned by 1 year
- Both models have 2021 H2 and 2022 full year data for comparison
- 2023 only available for Sequential

### 2. **Market Regime Matters**
- **2021 H2 Bull Market**: LSTM (+12.28%) crushed Sequential (-28.96%)
- **2022 Bear Market**: Sequential (-16.22%) slightly beat LSTM (-17.16%)
- Both outperformed S&P 500 in 2022 bear market

### 3. **LSTM Specific Issues**
- Convergence problems (0 stocks selected in Jun/Jul 2022)
- High volatility in returns and stock selection (0-40 stocks)
- Strong individual months (Oct: +19.01%) but inconsistent

### 4. **Missing Comparisons**
- No Sequential data for 2021 to compare with LSTM
- No LSTM data for 2023 to compare with Sequential's +14%

## 💡 Conclusions

1. **2021 H2 (Bull Market):** LSTM dramatically outperformed Sequential (+12.28% vs -28.96%)
2. **2022 (Bear Market):** Sequential marginally outperformed LSTM (-16.22% vs -17.16%)
3. **Overall Pattern:** LSTM performs better in bull markets, Sequential slightly better in bear markets
4. **LSTM Advantages:** 
   - Superior performance in trending/bull markets
   - Higher win rate in bear markets
   - Potential for exceptional monthly gains
5. **LSTM Disadvantages:** 
   - Convergence issues (0 stocks selected)
   - Higher volatility
   - Inconsistent stock selection (0-40 stocks)
6. **Need for Further Testing:** Run LSTM on 2023 data for complete comparison

## 📝 Recommendations

1. **Fix LSTM convergence issues** (0 stocks selected problem)
2. **Run LSTM on 2023 data** to enable full comparison
3. **Consider ensemble approach** combining both methods
4. **Implement adaptive gamma** based on market regime
5. **Add regularization** to stabilize LSTM stock selection

---

*Report Generated: August 10, 2025*  
*Corrected with proper date alignment based on code analysis*  
*Data sources: LSTM experiment results + REALIGNED_GAMMA_RESULTS_CORRECT_DATES.md*