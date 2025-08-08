"""
Investigate why models underperformed in 2021 bull market
"""

import json
import numpy as np

def analyze_2021_performance():
    """Analyze 2021 performance across all gammas"""
    
    print("2021 PERFORMANCE INVESTIGATION")
    print("="*80)
    print("\n1. MARKET CONTEXT (H2 2021):")
    print("-"*40)
    print("• S&P 500: +11% (Jul-Dec)")
    print("• Market Type: Strong Bull Market")
    print("• Key Events: Post-COVID recovery, Fed stimulus, Tech rally")
    print("• Volatility: Moderate to Low")
    
    print("\n2. MODEL PERFORMANCE COMPARISON:")
    print("-"*40)
    
    # Results from our experiments
    results = {
        'gamma_0.1': {
            'returns': [-17.49, -0.38, -0.79, -9.53, 9.90, -22.30],
            'annual': -37.00,
            'win_rate': 17
        },
        'gamma_0.3': {
            'returns': [-15.00, -0.55, -0.79, 2.09, 1.47, -20.88],
            'annual': -31.27,
            'win_rate': 33
        },
        'gamma_0.5': {
            'returns': [-19.53, -0.64, -0.67, 18.55, 1.55, -2.52],
            'annual': -6.80,
            'win_rate': 33
        }
    }
    
    months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print(f"{'Month':<6} | {'γ=0.1':>10} | {'γ=0.3':>10} | {'γ=0.5':>10} | {'S&P 500*':>10}")
    print("-"*60)
    
    # Approximate S&P monthly returns for H2 2021
    sp500_approx = [0.0, 2.5, -4.7, 6.9, -0.8, 4.4]  # Rough estimates
    
    for i, month in enumerate(months):
        g01 = results['gamma_0.1']['returns'][i]
        g03 = results['gamma_0.3']['returns'][i]
        g05 = results['gamma_0.5']['returns'][i]
        sp = sp500_approx[i]
        
        print(f"{month:<6} | {g01:>9.1f}% | {g03:>9.1f}% | {g05:>9.1f}% | {sp:>9.1f}%")
    
    print("-"*60)
    print(f"{'H2 2021':<6} | {results['gamma_0.1']['annual']:>9.1f}% | "
          f"{results['gamma_0.3']['annual']:>9.1f}% | "
          f"{results['gamma_0.5']['annual']:>9.1f}% | {'~11.0':>9}%")
    
    print("\n3. KEY OBSERVATIONS:")
    print("-"*40)
    
    # Calculate correlation with market
    for gamma_name, gamma_data in results.items():
        returns = gamma_data['returns']
        correlation = np.corrcoef(returns, sp500_approx)[0, 1]
        print(f"• {gamma_name}: Correlation with S&P = {correlation:.3f}")
    
    print("\n4. PROBLEMATIC PATTERNS:")
    print("-"*40)
    print("• July 2021: All models had massive losses (-15% to -19%)")
    print("  → S&P was flat/slightly up in July")
    print("• December 2021: All models lost heavily (-2.5% to -22%)")
    print("  → S&P gained +4.4% in December")
    print("• October 2021: Only bright spot (γ=0.5: +18.55%)")
    print("  → Aligned with S&P's +6.9%")
    
    print("\n5. POTENTIAL ISSUES:")
    print("-"*40)
    print("a) DATA QUALITY/ALIGNMENT:")
    print("   • Are we using the right files for 2021?")
    print("   • Files labeled 'test_data_start_date_2020' for 2021 trading")
    print("   • Only 6 months of data (incomplete training?)")
    
    print("\nb) TRAINING DATA ISSUES:")
    print("   • 2021 models trained on 2020 COVID crash data")
    print("   • Model may have learned crash patterns")
    print("   • Expecting volatility that didn't materialize")
    
    print("\nc) MARKET REGIME MISMATCH:")
    print("   • Models trained on high volatility (2020)")
    print("   • Applied to low volatility bull market (2021)")
    print("   • Gamma discounting may be suboptimal for this transition")
    
    print("\nd) STOCK SELECTION BIAS:")
    print("   • Our universe might differ from S&P 500")
    print("   • Could be picking wrong stocks")
    print("   • Transaction costs eating returns")
    
    print("\n6. HYPOTHESIS:")
    print("-"*40)
    print("The models likely failed because:")
    print("1. Training on 2020 COVID crash data created bearish bias")
    print("2. Only 7 consecutive files for training (limited context)")
    print("3. First half of 2021 data missing (Jan-Jun)")
    print("4. Models may be picking defensive stocks in bull market")
    
    print("\n7. RECOMMENDATIONS:")
    print("-"*40)
    print("1. Check if we have Jan-Jun 2021 data files")
    print("2. Verify stock universe and selection criteria")
    print("3. Consider market regime detection before gamma selection")
    print("4. Test with longer training sequences (>7 files)")
    print("5. Add benchmark comparison to track relative performance")

if __name__ == "__main__":
    analyze_2021_performance()