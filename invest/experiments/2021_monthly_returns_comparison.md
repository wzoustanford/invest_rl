# 2021 Monthly Returns - Correct Alignment with S&P 500 Comparison

## Important Note on Trading Periods
Each "monthly" trade actually spans ~25 days from buy to sell, so returns overlap months.

---

## 📊 2021 H2 Trading Results (What We Have)

### Monthly Returns Table

| Trading Period | File Date | γ=0.1 | γ=0.3 | γ=0.5 | S&P 500* | Best Model |
|----------------|-----------|--------|--------|--------|----------|------------|
| **Jul-Aug 2021** | 2020_07_01 | -17.68% | -12.26% | -10.19% | +2.3% | γ=0.5 (least loss) |
| **Aug-Sep 2021** | 2020_08_01 | -0.38% | -0.52% | -0.61% | +2.9% | γ=0.1 (least loss) |
| **Sep-Oct 2021** | 2020_09_01 | -0.79% | -0.79% | -0.69% | -4.7% | γ=0.5 (beat S&P) |
| **Oct-Nov 2021** | 2020_10_01 | -9.54% | +2.15% | +15.45% | +6.9% | γ=0.5 (beat S&P) |
| **Nov-Dec 2021** | 2020_11_01 | +9.94% | +1.47% | +1.35% | -0.7% | γ=0.1 (beat S&P) |
| **Dec21-Jan22** | 2020_12_01 | -22.47% | -20.86% | -15.83% | +4.5% | γ=0.5 (least loss) |

*S&P 500 returns are approximate for the corresponding periods

---

## 📈 S&P 500 Monthly Performance in 2021

### Full Year 2021 S&P 500 Returns (for context):

| Month | S&P 500 Return | Our Trading? | Notes |
|-------|----------------|--------------|-------|
| January | -1.0% | ❌ No data | - |
| February | +2.6% | ❌ No data | - |
| March | +4.2% | ❌ No data | - |
| April | +5.2% | ❌ No data | Would need 2019 files |
| May | +0.5% | ❌ No data | Would need 2019 files |
| June | +2.2% | ❌ No data | Would need early 2020 files |
| **July** | **+2.3%** | ✅ **Traded** | **We lost 10-18%** |
| **August** | **+2.9%** | ✅ **Traded** | **We lost 0.4-0.6%** |
| **September** | **-4.7%** | ✅ **Traded** | **We lost 0.7-0.8%** |
| **October** | **+6.9%** | ✅ **Traded** | **Mixed: -9.5% to +15.5%** |
| **November** | **-0.7%** | ✅ **Traded** | **We gained 1.4-9.9%** |
| **December** | **+4.5%** | ✅ **Traded** | **We lost 15-22%** |

**2021 Full Year S&P 500**: +26.9%  
**2021 H2 (Jul-Dec) S&P 500**: ~+11%

---

## 🔍 Performance Analysis

### Correlation with Market (H2 2021)

| Strategy | Correlation with S&P | Win Rate | Avg Return | Total H2 Return |
|----------|---------------------|----------|------------|-----------------|
| γ=0.1 | -0.458 | 17% (1/6) | -6.82% | -37.27% |
| γ=0.3 | -0.148 | 33% (2/6) | -5.13% | -28.96% |
| γ=0.5 | +0.489 | 33% (2/6) | -1.75% | -12.70% |
| **S&P 500** | 1.000 | 67% (4/6) | +1.8% | **+11%** |

---

## 🚨 Key Observations

### Major Disconnects:
1. **July 2021 Disaster**: S&P +2.3%, We lost 10-18%
2. **December 2021 Collapse**: S&P +4.5%, We lost 15-22%
3. **August 2021**: S&P +2.9%, We lost ~0.5%

### Partial Successes:
1. **October 2021**: γ=0.5 gained +15.45% vs S&P +6.9% ✅
2. **November 2021**: All models beat S&P (which was -0.7%)
3. **September 2021**: Losses smaller than S&P's -4.7%

---

## 💡 Root Cause Analysis

### Why Did We Fail in H2 2021 Bull Market?

1. **Training Data Issue**:
   - July 2021 model trained on: June 2020 - June 2021 data
   - This includes the COVID crash and volatile recovery
   - Model learned defensive patterns inappropriate for steady bull market

2. **Missing H1 2021 Context**:
   - We lack Jan-Jun 2021 trading
   - H1 2021 S&P gained +14.4%
   - Models never saw this steady uptrend

3. **Regime Change Blindness**:
   - 2020: COVID crash → volatile recovery
   - 2021: Steady bull market with low volatility
   - Models expected continued volatility that never came

4. **Negative Correlation**:
   - γ=0.1 had -0.458 correlation with S&P!
   - Models were effectively betting AGAINST the market

---

## 📊 Summary Statistics

### 2021 H2 Performance Summary

| Metric | γ=0.1 | γ=0.3 | γ=0.5 | S&P 500 |
|--------|-------|-------|-------|---------|
| **Total Return** | -37.27% | -28.96% | -12.70% | +11% |
| **Best Month** | +9.94% (Nov) | +2.15% (Oct) | +15.45% (Oct) | +6.9% (Oct) |
| **Worst Month** | -22.47% (Dec) | -20.86% (Dec) | -15.83% (Dec) | -4.7% (Sep) |
| **Months Positive** | 1/6 | 2/6 | 2/6 | 4/6 |
| **Avg Monthly** | -6.82% | -5.13% | -1.75% | +1.8% |
| **Underperformance** | -48.3% | -40.0% | -23.7% | - |

---

## 🎯 Conclusion

With correct date alignment, it's clear that:
1. We only traded H2 2021 (July-December)
2. All strategies massively underperformed the S&P 500
3. The models had negative or near-zero correlation with the market
4. Training on COVID-era data created inappropriate defensive bias
5. December 2021 was particularly disastrous (lost 15-22% when S&P gained 4.5%)

The 2021 results highlight a fundamental flaw: models trained on crisis/recovery data (2020-early 2021) completely failed to adapt to the steady bull market conditions of H2 2021.

---

*Data Source: Experimental results from gamma experiments*  
*S&P 500 data: Approximate monthly returns for 2021*  
*Trading periods: ~25 days each starting from indicated month*