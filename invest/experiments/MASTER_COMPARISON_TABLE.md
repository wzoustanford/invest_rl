# 📊 MASTER COMPARISON TABLE - All Models & Experiments
## Last Updated: August 10, 2025

### 🎯 Primary Reference Document
This is the master comparison table for all trading model experiments. All updates should be made here.

### 📁 Key Files:
- **This file**: `/home/ubuntu/code/angle_rl/invest/experiments/MASTER_COMPARISON_TABLE.md`
- **Detailed LSTM results**: `/home/ubuntu/code/angle_rl/invest/experiments/lstm_750iter_20250810_054909/comparison_report_750iter_CORRECTED_DATES.md`
- **Sequential results source**: `/home/ubuntu/code/angle_rl/invest/experiments/REALIGNED_GAMMA_RESULTS_CORRECT_DATES.md`

---

## ⚠️ CRITICAL: Date Alignment
- Files with `test_data_start_date_2020_XX_XX` → Trade in **2021**
- Files with `test_data_start_date_2021_XX_XX` → Trade in **2022**
- Files with `test_data_start_date_2022_XX_XX` → Trade in **2023**
- Trading happens ~360 days AFTER test_data_start_date

---

## 📈 MASTER RESULTS TABLE

| Year | Period | Model | Config | Annual Return | vs S&P | Win Rate | Best Month | Worst Month | Status |
|------|--------|-------|--------|---------------|--------|----------|------------|-------------|---------|
| **2021 H2** | Jul-Dec | Sequential | γ=0.3, 750it | -28.96% | -39.96pp | - | - | - | ✅ Complete |
| **2021 H2** | May-Dec | LSTM | γ=0.3, 750it | **+12.28%** | +1.28pp | 50% | +12.34% | -4.87% | ✅ Complete |
| | | | | | | | | | |
| **2022** | Jan-Dec | Sequential | γ=0.3, 750it | **-16.22%** | +1.88pp | 33% | +16.14% | -18.64% | ✅ Complete |
| **2022** | Jan-Dec | LSTM | γ=0.3, 750it | -17.16% | +0.94pp | 42% | +19.01% | -21.59% | ✅ Complete |
| | | | | | | | | | |
| **2023** | Jan-Dec | Sequential | γ=0.3, 750it | +14.00% | -10.2pp | 42% | +56.98% | -16.19% | ✅ Complete |
| **2023** | Jan-Dec | LSTM | γ=0.3, 750it | - | - | - | - | - | ❌ Not Run |
| | | | | | | | | | |
| **2024** | Jan-Dec | Sequential | γ=0.3, 750it | +406.17% | +382.9pp | 75% | +81.19% | -9.24% | ✅ Complete |
| **2024** | Jan-Dec | LSTM | γ=0.3, 750it | - | - | - | - | - | ❌ Not Run |

### Market Performance Reference:
- **2021 H2**: S&P 500 +11%
- **2022**: S&P 500 -18.1%
- **2023**: S&P 500 +24.2%
- **2024**: S&P 500 +23.3%

---

## 🏆 Head-to-Head Winners

| Year | Winner | Margin | Key Factor |
|------|--------|--------|------------|
| **2021 H2** | LSTM | +41.24pp | Bull market outperformance |
| **2022** | Sequential | +0.94pp | Bear market stability |
| **2023** | Sequential* | N/A | *LSTM not tested |
| **2024** | Sequential* | N/A | *LSTM not tested |

---

## 📊 Model Characteristics

### LSTM (750 iterations, γ=0.3)
- **Strengths**: 
  - Excellent bull market performance
  - Potential for exceptional monthly gains
  - Higher win rate in some conditions
- **Weaknesses**:
  - Convergence issues (0 stocks selected)
  - High volatility (0-40 stocks)
  - Inconsistent performance

### Sequential Supervised (750 iterations, γ=0.3)
- **Strengths**:
  - Consistent across market conditions
  - Better bear market protection
  - Spectacular 2024 performance
- **Weaknesses**:
  - Poor 2021 H2 performance
  - Lower ceiling in normal conditions

---

## 🔬 Known Issues & TODOs

### Issues:
1. **LSTM Convergence**: Zero stocks selected in some months (Jun/Jul 2022)
2. **Date Alignment**: Must always verify trading year is correct
3. **Incomplete Testing**: LSTM not run on 2023-2024 data

### TODOs:
- [ ] Run LSTM on 2023 data
- [ ] Run LSTM on 2024 data
- [ ] Fix LSTM convergence issues
- [ ] Test with different gamma values
- [ ] Implement ensemble approach

---

## 📝 Update Log

| Date | Update | By |
|------|--------|-----|
| 2025-08-10 | Created master table with corrected date alignments | System |
| 2025-08-10 | Added 2021 H2 Sequential data (-28.96%) | System |
| 2025-08-10 | Confirmed LSTM 2021-2022 results with 750 iterations | System |

---

## 🎯 Next Steps

1. **Priority 1**: Run LSTM on 2023 data for direct comparison
2. **Priority 2**: Fix LSTM convergence issues (0 stocks problem)
3. **Priority 3**: Run LSTM on 2024 data to verify if Sequential's +406% is reproducible

---

*This is the authoritative comparison document. All future experiments should update this table.*