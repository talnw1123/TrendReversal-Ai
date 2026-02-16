import pandas as pd
import numpy as np

def analyze_regimes():
    file_path = 'all_trad/regime_comparison_results.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # Calculate Overall Accuracy
    df['overall_accuracy'] = (df['uptrend_accuracy'] + df['downtrend_accuracy']) / 2
    
    # Get unique markets
    markets = df['Market'].unique()
    
    print("=" * 80)
    print("REGIME DETECTION ACCURACY ANALYSIS")
    print("=" * 80)
    
    for market in markets:
        print(f"\n>>> MARKET: {market}")
        market_df = df[df['Market'] == market].copy()
        
        # Sort by Overall Accuracy
        print("\n  [Ranked by Accuracy (Correctly Identified Days)]")
        acc_sorted = market_df.sort_values(by='overall_accuracy', ascending=False)
        print(acc_sorted[['Method', 'overall_accuracy', 'uptrend_accuracy', 'downtrend_accuracy']].to_string(index=False))
        
        # Sort by Separation (Quality of Returns)
        print("\n  [Ranked by Market Separation (Good Uptrend vs Bad Downtrend)]")
        # Separation = Avg Uptrend Return - Avg Downtrend Return
        # We want Positive Separation (Uptrends are green, Downtrends are red)
        sep_sorted = market_df.sort_values(by='separation', ascending=False)
        print(sep_sorted[['Method', 'separation', 'avg_uptrend_ret', 'avg_downtrend_ret']].to_string(index=False))
        
        # Recommendation
        best_acc_method = acc_sorted.iloc[0]['Method']
        best_sep_method = sep_sorted.iloc[0]['Method']
        
        print(f"\n  🏆 Champion based on Accuracy: {best_acc_method} ({acc_sorted.iloc[0]['overall_accuracy']:.2f}%)")
        print(f"  💎 Champion based on Separation: {best_sep_method} (Sep: {sep_sorted.iloc[0]['separation']:.4f})")
        
        if best_acc_method != best_sep_method:
             print(f"  ⚠️ Note: High accuracy doesn't always mean profitable. Check Separation.")

    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_regimes()
