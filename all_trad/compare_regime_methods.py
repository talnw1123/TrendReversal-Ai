"""
Compare Regime Detection Methods
Evaluates: GMM, ADX+Supertrend, HMM, SMA200
Measures how well each method identifies profitable trends
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../model'))

from config import TICKERS
from model.regime_detection import RegimeDetector


def calculate_trend_returns(df, is_uptrend):
    """Calculate returns during uptrend vs downtrend periods"""
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['is_uptrend'] = is_uptrend
    
    # Next-day returns (we predict today, returns are tomorrow)
    df['next_returns'] = df['returns'].shift(-1)
    
    # Clean NaN
    df = df.dropna()
    
    # Split by trend
    uptrend_mask = df['is_uptrend'] == 1
    downtrend_mask = df['is_uptrend'] == 0
    
    # Calculate metrics
    uptrend_returns = df.loc[uptrend_mask, 'next_returns']
    downtrend_returns = df.loc[downtrend_mask, 'next_returns']
    
    # Strategy return: Long during uptrend, Cash/Short during downtrend
    # Simple: Long uptrend only
    strategy_returns = df['next_returns'].copy()
    strategy_returns[downtrend_mask] = 0  # Stay in cash during downtrend
    
    # Calculate cumulative
    cum_strategy = (1 + strategy_returns).cumprod().iloc[-1] - 1
    cum_buyhold = (1 + df['next_returns']).cumprod().iloc[-1] - 1
    
    # Accuracy: How often uptrend has positive returns?
    uptrend_accuracy = (uptrend_returns > 0).mean() if len(uptrend_returns) > 0 else 0
    downtrend_accuracy = (downtrend_returns < 0).mean() if len(downtrend_returns) > 0 else 0
    
    # Trend detection quality
    # Good detection: uptrend has higher avg returns than downtrend
    avg_uptrend = uptrend_returns.mean() if len(uptrend_returns) > 0 else 0
    avg_downtrend = downtrend_returns.mean() if len(downtrend_returns) > 0 else 0
    separation = avg_uptrend - avg_downtrend
    
    # Calculate Max Drawdown for strategy
    cum_ret = (1 + strategy_returns).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = cum_ret / rolling_max - 1
    max_dd = drawdown.min() * 100
    
    # Calculate Sharpe Ratio for strategy (annualized)
    # Assuming daily data, 252 trading days
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    
    return {
        'uptrend_days': len(uptrend_returns),
        'downtrend_days': len(downtrend_returns),
        'uptrend_pct': len(uptrend_returns) / len(df) * 100,
        'avg_uptrend_ret': avg_uptrend * 100,
        'avg_downtrend_ret': avg_downtrend * 100,
        'separation': separation * 100,
        'uptrend_accuracy': uptrend_accuracy * 100,
        'downtrend_accuracy': downtrend_accuracy * 100,
        'strategy_return': cum_strategy * 100,
        'buyhold_return': cum_buyhold * 100,
        'excess_return': (cum_strategy - cum_buyhold) * 100,
        'max_dd': max_dd,
        'sharpe': sharpe
    }


def compare_methods():
    """Compare all regime detection methods across all markets"""
    
    results = []
    
    print("=" * 80)
    print("REGIME DETECTION METHODS COMPARISON")
    print("=" * 80)
    
    for market, ticker in TICKERS.items():
        print(f"\n--- {market} ({ticker}) ---")
        
        # Define methods inside loop to access 'market' variable
        methods = {
            'SMA200': lambda df: (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values,
            'GMM': lambda df: RegimeDetector.detect_gmm(df),
            'GMM_Enhanced': lambda df: RegimeDetector.detect_gmm_enhanced(df),
            'ADX_Supertrend': lambda df: RegimeDetector.detect_adx_supertrend(df),
            'HMM': lambda df: RegimeDetector.detect_hmm(df),
            'HMM_Enhanced': lambda df: RegimeDetector.detect_hmm_enhanced(df),
            'RandomForest': lambda df: RegimeDetector.detect_random_forest(df, market_name=market)
        }
        
        # Fetch data
        try:
            df = yf.download(ticker, period="max", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if df.empty or len(df) < 250:
                print(f"  Skipping {market}: Not enough data")
                continue
            
            print(f"  Data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
            
            for method_name, method_func in methods.items():
                try:
                    is_uptrend = method_func(df)
                    metrics = calculate_trend_returns(df, is_uptrend)
                    
                    result = {
                        'Market': market,
                        'Method': method_name,
                        **metrics
                    }
                    results.append(result)
                    
                    print(f"  {method_name:15} | Uptrend: {metrics['uptrend_pct']:5.1f}% | "
                          f"Sep: {metrics['separation']:+6.3f}% | "
                          f"Strategy: {metrics['strategy_return']:+7.1f}% | "
                          f"B&H: {metrics['buyhold_return']:+7.1f}% | "
                          f"Excess: {metrics['excess_return']:+6.1f}%")
                    
                except Exception as e:
                    print(f"  {method_name}: Error - {e}")
                    
        except Exception as e:
            print(f"  Error processing {market}: {e}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        
        # Save raw results
        output_path = os.path.join(os.path.dirname(__file__), 'regime_comparison_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        print("\n" + "=" * 80)
        print("SUMMARY: BEST METHOD PER MARKET")
        print("=" * 80)
        
        markets = results_df['Market'].unique()
        
        summary_list = []
        
        for market in markets:
            print(f"\n>>> MARKET: {market}")
            market_df = results_df[results_df['Market'] == market].copy()
            
            # Sort by Separation (Higher is better)
            market_df.sort_values('separation', ascending=False, inplace=True)
            
            # Print ranked methods
            print(market_df[['Method', 'separation', 'uptrend_accuracy', 'excess_return']].to_string(index=False))
            
            best_method = market_df.iloc[0]
            print(f"  🏆 WINNER: {best_method['Method']} (Sep: {best_method['separation']:.3f}%)")
            
            summary_list.append({
                'Market': market,
                'Best_Method': best_method['Method'],
                'Separation': best_method['separation'],
                'Excess_Return': best_method['excess_return']
            })
            
        summary_df = pd.DataFrame(summary_list)
        return results_df, summary_df
    
    return None, None


if __name__ == "__main__":
    compare_methods()
