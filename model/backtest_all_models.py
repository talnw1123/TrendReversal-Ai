"""
Backtest all models on all markets and trends.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Parameters
LOOKBACK = 60
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001

MARKETS = ['US', 'UK', 'Thai', 'Gold', 'BTC']
TRENDS = ['uptrend', 'downtrend']
KERAS_MODELS = ['LSTM', 'CNN', 'MLP', 'Transformer']
ML_MODELS = ['RandomForest', 'SVM']

def calculate_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min() * 100

def load_market_test_data(market, trend):
    """Load and prepare test data for a market/trend with warmup."""
    from features import calculate_features, get_feature_columns
    
    test_file = glob.glob(f"../trend_data_manual/split/test/**/{market}_{trend}_labeled.csv", recursive=True)
    if not test_file:
        return None, None, None, None
    
    df_test = pd.read_csv(test_file[0], index_col=0, parse_dates=True).sort_index()
    test_start_date = df_test.index.min()
    
    # Warmup with validation data
    val_file = glob.glob(f"../trend_data_manual/split/val/**/{market}_{trend}_labeled.csv", recursive=True)
    if val_file:
        df_val = pd.read_csv(val_file[0], index_col=0, parse_dates=True).sort_index()
        df_combined = pd.concat([df_val, df_test])
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')].sort_index()
    else:
        df_combined = df_test
    
    df_combined = calculate_features(df_combined)
    if df_combined.empty:
        return None, None, None, None
    
    feature_cols = get_feature_columns(df_combined)
    data = df_combined[feature_cols].values
    prices = df_combined['Close'].values
    dates = df_combined.index
    
    # Extract valid test windows
    X_list = []
    valid_indices = []
    
    for i in range(LOOKBACK, len(data)):
        if dates[i] >= test_start_date:
            X_list.append(data[i-LOOKBACK:i])
            valid_indices.append(i)
    
    if not X_list:
        return None, None, None, None
    
    return np.array(X_list), prices[valid_indices], dates[valid_indices], len(feature_cols)

def run_simulation(signals, prices):
    """Run trading simulation."""
    cash = INITIAL_CAPITAL
    position = 0
    equity_curve = []
    
    for i in range(len(signals) - 1):
        signal = signals[i]
        price = prices[i]
        next_price = prices[i+1]
        
        target_pos = 0
        if signal == 2: target_pos = 1
        elif signal == 0: target_pos = -1
        
        if position != target_pos:
            cost = cash * abs(target_pos - position) * TRANSACTION_COST
            cash -= cost
            position = target_pos
        
        if position == 1:
            cash *= (1 + (next_price - price) / price)
        elif position == -1:
            cash *= (1 + (price - next_price) / price)
        
        equity_curve.append(cash)
    
    return equity_curve

def main():
    print("=" * 70)
    print("BACKTEST ALL MODELS")
    print("=" * 70)
    
    # Load training data for scaler
    print("\nFitting scaler...")
    from train_reversal_model import load_and_process_data
    X_train, _ = load_and_process_data('train')
    N, T, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    
    # Load all models
    models = {}
    for name in KERAS_MODELS:
        path = f"reversal_model_{name}.keras"
        if os.path.exists(path):
            models[name] = ('keras', load_model(path))
            print(f"  Loaded {name}")
    
    for name in ML_MODELS:
        path = f"reversal_model_{name}.pkl"
        if os.path.exists(path):
            models[name] = ('ml', joblib.load(path))
            print(f"  Loaded {name}")
    
    results = []
    
    for market in MARKETS:
        for trend in TRENDS:
            print(f"\n{'='*50}")
            print(f"Market: {market} | Trend: {trend}")
            print(f"{'='*50}")
            
            X_test, prices, dates, num_features = load_market_test_data(market, trend)
            
            if X_test is None or len(X_test) == 0:
                print("  No data available")
                continue
            
            # Scale
            X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
            
            # Buy & Hold
            bnh_return = (prices[-1] - prices[0]) / prices[0] * 100
            
            for model_name, (model_type, model) in models.items():
                try:
                    if model_type == 'keras':
                        probs = model.predict(X_test_scaled, verbose=0)
                        signals = np.argmax(probs, axis=1)
                    else:
                        X_2d = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
                        signals = model.predict(X_2d)
                    
                    equity_curve = run_simulation(signals, prices)
                    
                    if not equity_curve:
                        continue
                    
                    final_equity = equity_curve[-1]
                    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                    equity_series = pd.Series(equity_curve)
                    max_dd = calculate_drawdown(equity_series)
                    
                    results.append({
                        'Market': market,
                        'Trend': trend,
                        'Model': model_name,
                        'Return (%)': round(total_return, 2),
                        'B&H (%)': round(bnh_return, 2),
                        'Max DD (%)': round(max_dd, 2),
                        'Final $': round(final_equity, 2)
                    })
                    
                    print(f"  {model_name}: {total_return:+.2f}% (B&H: {bnh_return:+.2f}%)")
                    
                except Exception as e:
                    print(f"  {model_name}: Error - {e}")
    
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("backtest_all_models.csv", index=False)
    
    # Print Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BEST MODEL PER MARKET/TREND")
    print("=" * 70)
    
    for market in MARKETS:
        for trend in TRENDS:
            subset = results_df[(results_df['Market'] == market) & (results_df['Trend'] == trend)]
            if len(subset) > 0:
                best = subset.loc[subset['Return (%)'].idxmax()]
                print(f"{market} {trend}: {best['Model']} ({best['Return (%)']}%)")
    
    print(f"\nFull results saved to backtest_all_models.csv")

if __name__ == "__main__":
    main()
