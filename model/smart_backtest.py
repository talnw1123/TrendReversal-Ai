"""
Smart Backtest: Uses SMA200 to switch between uptrend and downtrend models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier
from regime_detection import RegimeDetector

# Parameters
LOOKBACK = 30
SMA_PERIOD = 200
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001
THRESHOLD_LONG = 0.54
THRESHOLD_SHORT = 0.46

MARKETS = ['US', 'UK', 'Thai', 'Gold', 'BTC']

def load_full_market_data(market):
    """Load full price history for a market (train + val + test)."""
    from features import calculate_features, get_selected_features
    
    all_data = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for split in ['train', 'val', 'test']:
        for trend in ['uptrend', 'downtrend']:
            # Use absolute path relative to script location
            base_dir = os.path.join(script_dir, "../trend_data_manual/split", split)
            pattern = os.path.join(base_dir, "**", f"{market}_{trend}_labeled.csv")
            files = glob.glob(pattern, recursive=True)
            for f in files:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                all_data.append(df)
    
    if not all_data:
        return None
    
    # Combine and sort
    df_combined = pd.concat(all_data)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')].sort_index()
    
    return df_combined

def prepare_features(df):
    """Apply feature engineering."""
    from features import calculate_features, get_selected_features
    
    df = calculate_features(df)
    if df.empty:
        return None, None
    
    feature_cols = get_selected_features(df)
    return df, feature_cols

    feature_cols = get_selected_features(df)
    return df, feature_cols

def plot_trades(market, model_name, dates, prices, events, final_return, bh_return):
    """Plot price history and trade markers."""
    plt.figure(figsize=(14, 7))
    plt.plot(dates, prices, label='Price', color='black', alpha=0.6, linewidth=1)
    
    # Process events
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    
    for date, action, price in events:
        if action == 'Buy':
            buy_dates.append(date)
            buy_prices.append(price)
        elif action == 'Sell':
            sell_dates.append(date)
            sell_prices.append(price)
            
    if buy_dates:
        plt.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
    if sell_dates:
        plt.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell', zorder=5)
        
    plt.title(f"{market} - {model_name}\nStrategy: {final_return:+.2f}% vs B&H: {bh_return:+.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    save_dir = "backtest_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, f"{market}_{model_name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"  Saved plot to {filename}")

def run_smart_backtest():
    print("=" * 70)
    print("SMART BACKTEST: SMA200 MODEL SWITCHING")
    print("=" * 70)
    
    # Load scalers (fit on training data for each trend)
    print("\nFitting scalers...")
    from train_separate_models import load_trend_data
    
    # Uptrend Scaler
    print("  Loading Uptrend data...")
    X_train_up, _, _, _, _, _, _ = load_trend_data('uptrend')
    N, T, F = X_train_up.shape
    scaler_up = StandardScaler()
    scaler_up.fit(X_train_up.reshape(-1, F))
    
    # Downtrend Scaler
    print("  Loading Downtrend data...")
    X_train_down, _, _, _, _, _, _ = load_trend_data('downtrend')
    scaler_down = StandardScaler()
    scaler_down.fit(X_train_down.reshape(-1, F))
    
    # Load separate models
    print("\nLoading models and calibrating...")
    from sklearn.isotonic import IsotonicRegression
    
    uptrend_models = {}
    downtrend_models = {}
    calibrators = {} # Key: (model_name, trend) -> IsotonicRegression
    
    # Load Validation Data for Calibration
    print("  Loading validation data for calibration...")
    # Helper to load val data specifically (reusing load_trend_data logic would be cleaner but let's do it direct for speed)
    X_val_up, y_val_up, _, _, _, _, _ = load_trend_data('uptrend') 
    # Wait, load_trend_data returns (train, val, test). We need the 2nd and 4th elements.
    # Actually load_trend_data returns: 
    # (X_train, y_train, X_val, y_val, X_test, y_test, market_test_data)
    _, _, X_val_up, y_val_up, _, _, _ = load_trend_data('uptrend')
    _, _, X_val_down, y_val_down, _, _, _ = load_trend_data('downtrend')
    
    # Scale Validation Data
    # Note: We must use the SAME scalers fitted on Train
    N_vu, T_vu, F_vu = X_val_up.shape
    X_val_up_scaled = scaler_up.transform(X_val_up.reshape(-1, F)).reshape(N_vu, T_vu, F_vu)
    X_val_up_scaled = np.nan_to_num(X_val_up_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    N_vd, T_vd, F_vd = X_val_down.shape
    X_val_down_scaled = scaler_down.transform(X_val_down.reshape(-1, F)).reshape(N_vd, T_vd, F_vd)
    X_val_down_scaled = np.nan_to_num(X_val_down_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    for model_name in ['LSTM', 'CNN', 'MLP', 'Transformer', 'RandomForest', 'SVM', 'XGBoost']:
        # --- UPTREND MODEL ---
        up_path_keras = f"model_uptrend/{model_name}.keras"
        up_path_pkl = f"model_uptrend/{model_name}.pkl"
        
        up_model = None
        if os.path.exists(up_path_keras):
            up_model = load_model(up_path_keras)
        elif os.path.exists(up_path_pkl):
            up_model = joblib.load(up_path_pkl)
            
        if up_model:
            uptrend_models[model_name] = up_model
            # Calibrate
            print(f"    Calibrating {model_name} (Uptrend)...")
            if model_name in ['RandomForest', 'SVM', 'XGBoost']:
                probs_val = up_model.predict_proba(X_val_up_scaled.reshape(len(X_val_up), -1))
            else:
                probs_val = up_model.predict(X_val_up_scaled, verbose=0)
            
            # Binary Mode: Prob of class 1
            prob_pos = probs_val[:, 1]
            
            # Determine mapping for Isotonic
            # If BINARY_MODE is ON: y_val_up is 0, 1.
            # If BINARY_MODE is OFF: y_val_up is -1, 0, 1. (But separate_models said it returns -1,0,1 or 0,1 based on flag)
            # Let's assume prediction is always 0,1 (binary mode in training).
            # If multiclass, we need prob of Bullish.
            
            # For simplicity, assume labels are 0,1 (or we map -1->0, 1->1 if needed).
            # load_trend_data logic: 
            # If BINARY_MODE=True inside training script, it returns 0,1.
            # Here we called load_trend_data from THIS script? No, we imported it.
            # smart_backtest doesn't define BINARY_MODE constant, but train_separate_models does.
            # Let's verify y_val range.
            y_unique = np.unique(y_val_up)
            # print(f"Unique validation labels: {y_unique}")
            
            # If labels are -1, 1: map to 0, 1 for isotonic
            y_calib = np.where(y_val_up == -1, 0, y_val_up)
            y_calib = np.where(y_calib == 1, 1, y_calib) 
            # If labels are 0, 1 (from binary mode), this is safe (0->0, 1->1)
            # If labels are -1, 0, 1: filter out 0? 
            # Ideally we only use samples with clear direction.
            
            iso_up = IsotonicRegression(out_of_bounds='clip')
            iso_up.fit(prob_pos, y_calib)
            calibrators[(model_name, 'uptrend')] = iso_up

        # --- DOWNTREND MODEL ---
        down_path_keras = f"model_downtrend/{model_name}.keras"
        down_path_pkl = f"model_downtrend/{model_name}.pkl"
        
        down_model = None
        if os.path.exists(down_path_keras):
            down_model = load_model(down_path_keras)
        elif os.path.exists(down_path_pkl):
            down_model = joblib.load(down_path_pkl)
            
        if down_model:
            downtrend_models[model_name] = down_model
            # Calibrate
            print(f"    Calibrating {model_name} (Downtrend)...")
            if model_name in ['RandomForest', 'SVM', 'XGBoost']:
                probs_val = down_model.predict_proba(X_val_down_scaled.reshape(len(X_val_down), -1))
            else:
                probs_val = down_model.predict(X_val_down_scaled, verbose=0)
                
            prob_pos = probs_val[:, 1]
            y_calib = np.where(y_val_down == -1, 0, y_val_down)
            y_calib = np.where(y_calib == 1, 1, y_calib)
            
            iso_down = IsotonicRegression(out_of_bounds='clip')
            iso_down.fit(prob_pos, y_calib)
            calibrators[(model_name, 'downtrend')] = iso_down
    
    results = []
    backtest_data = {} # Key: (Market, Model), Value: dict set
    
    for market in MARKETS:
        print(f"\n{'='*50}")
        print(f"Market: {market}")
        print(f"{'='*50}")
        
        # Load full market data
        df_raw = load_full_market_data(market)
        if df_raw is None:
            print("  No data available")
            continue
        
        # Calculate SMA200 for regime detection
        df_raw['SMA200'] = df_raw['Close'].rolling(window=SMA_PERIOD).mean()
        df_raw['is_uptrend'] = (df_raw['Close'] > df_raw['SMA200']).astype(int)
        
        # Apply features
        df, feature_cols = prepare_features(df_raw)
        if df is None:
            print("  Feature calculation failed")
            continue
        
        # Identify test period (last 15%)
        test_start_idx = int(len(df) * 0.85)
        test_df = df.iloc[test_start_idx:]
        
        if len(test_df) <= LOOKBACK:
            print("  Not enough test data")
            continue
        
        if len(test_df) <= LOOKBACK:
            print("  Not enough test data")
            continue
        
        # Define Regime Methods to test
        REGIME_METHODS = ['SMA200', 'GMM', 'ADX', 'HMM']
        
        for regime_name in REGIME_METHODS:
            print(f"\n  --- Regime Method: {regime_name} ---")
            
            # Calculate Regime
            if regime_name == 'SMA200':
                # Original Logic
                # Ensure we use df_raw (including train history for SMA calc)
                # But our df variable is 'df' (which is prepare_features(df_raw))
                # Wait, 'df' has features but we didn't store SMA200 in it?
                # prepare_features returns NEW df from calculate_features?
                # Ah, calculate_features takes df and ADDS cols.
                # In previous code:
                # df_raw['SMA200'] = ...
                # df_raw['is_uptrend'] = ...
                # these cols might be lost if calculate_features creates fresh copy or drops cols.
                # Let's check 'df' after prepare_features.
                # Actually, our regime detectors take 'df' (which has Close).
                # But for SMA200 we need history before test_start_idx.
                # We can compute is_uptrend on FULL df, then slice.
                
                # Re-compute SMA200 on 'df' (which is full history with features)
                full_is_uptrend = (df['Close'] > df['Close'].rolling(window=SMA_PERIOD).mean()).astype(int)
                
            elif regime_name == 'GMM':
                full_is_uptrend = RegimeDetector.detect_gmm(df)
            elif regime_name == 'ADX':
                full_is_uptrend = RegimeDetector.detect_adx_supertrend(df)
            elif regime_name == 'HMM':
                full_is_uptrend = RegimeDetector.detect_hmm(df)
            
            # Slice for test period
            # is_uptrend is array or series
            if isinstance(full_is_uptrend, pd.Series):
                 full_is_uptrend = full_is_uptrend.values
                 
            # We need to subset 'is_uptrend' to match 'data' (which is df[feature_cols])
            # But wait, the loop uses 'is_uptrend[i]' where i is index into 'data'.
            # 'data' comes from df[feature_cols].values.
            # So is_uptrend should be same length as 'data'.
            is_uptrend = full_is_uptrend
            
            # Prepare data and run simulation for THIS regime
            # (Variables need to be reset per regime loop?)
            # NO, data/prices/dates are same. But the LOOP over models needs to be nested inside regime?
            # Or regime inside model?
            # User wants to select best model. And now best regime?
            # Currently structure: For Market: -> For Model: -> Simulation.
            # We should change to: For Market: -> For Regime: -> For Model: -> Simulation.
            
            # So this replacement chunk is essentially wrapping the model loop.

            # Prepare data and run simulation
            data = df[feature_cols].values
            prices = df['Close'].values
            # is_uptrend = df['is_uptrend'].values # OLD
            
            dates = df.index
            
            for model_name in ['LSTM', 'CNN', 'MLP', 'Transformer', 'RandomForest', 'SVM', 'XGBoost']:
                if model_name not in uptrend_models or model_name not in downtrend_models:
                    continue
                
                daily_returns = []
                trend_status = [] # 1 for Uptrend, 0 for Downtrend
                cash = INITIAL_CAPITAL
                position = 0
                equity_curve = []
                trade_dates = []
                asset_returns = []
                
                # --- Audit Variables ---
                position_history = []
                trade_count = 0
                trade_events = [] # List of (date, action, price)
                
                for i in range(test_start_idx, len(data) - 1):
                    if i < LOOKBACK:
                        continue
                    
                    # Get current window
                    X_window = data[i-LOOKBACK:i].reshape(1, LOOKBACK, -1)
                    
                    # Report trend
                    current_is_uptrend = is_uptrend[i]
                    trend_status.append('Uptrend' if current_is_uptrend else 'Downtrend')
    
                    # Select model and scaler based on SMA200 regime
                    trend_key = 'uptrend' if current_is_uptrend else 'downtrend'
                    
                    if current_is_uptrend:
                        model = uptrend_models[model_name]
                        scaler = scaler_up
                    else:
                        model = downtrend_models[model_name]
                        scaler = scaler_down
                    
                    X_scaled = scaler.transform(X_window.reshape(-1, F)).reshape(X_window.shape)
                    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Predict Raw Probability
                    if model_name in ['RandomForest', 'SVM', 'XGBoost']:
                        # Sklearn models expect 2D input (flattened)
                        X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
                        probs = model.predict_proba(X_flat)
                    else:
                        # Keras models expect 3D input
                        probs = model.predict(X_scaled, verbose=0)
                    
                    # Raw probability of Bullish (Class 1)
                    raw_bull_prob = probs[0, 1]
                    
                    # Calibrate Probability
                    calibrator = calibrators.get((model_name, trend_key))
                    if calibrator:
                        cal_prob = calibrator.transform([raw_bull_prob])[0]
                    else:
                        cal_prob = raw_bull_prob
                    
                    # Threshold Logic with Hysteresis
                    # T_long = THRESHOLD_LONG, T_short = THRESHOLD_SHORT
                    # If prob >= T_long -> Long (1)
                    # If prob <= T_short -> Short (-1)
                    # Else -> Hold Current Position (Hysteresis)
                    
                    target_pos = position # Default to hold
                    
                    if cal_prob >= THRESHOLD_LONG:
                        target_pos = 1
                    elif cal_prob <= THRESHOLD_SHORT:
                        target_pos = -1
                    
                    # If current position is 0 (Flat) and we are in neutral zone, stay flat.
                    # If current position is 1 and prob drops to 0.5, stay 1 (until <= 0.4).
                    
                    # Transaction cost & Trade Counting
                    if position != target_pos:
                        cost = cash * abs(target_pos - position) * TRANSACTION_COST
                        cash -= cost
                        position = target_pos
                        trade_count += 1
                        
                        # Record Event
                        action = 'Buy' if target_pos == 1 else ('Sell' if target_pos == -1 else 'Exit')
                        trade_events.append((dates[i], action, prices[i]))
                    
                    # Audit: Track Position
                    position_history.append(position)
                    
                    price = prices[i]
                    next_price = prices[i + 1]
                    
                    # PnL
                    start_cash = cash
                    if position == 1:
                        cash *= (1 + (next_price - price) / price)
                    elif position == -1:
                        cash *= (1 + (price - next_price) / price)
                    
                    asset_ret = (next_price - price) / price
                    asset_returns.append(asset_ret)
    
                    ret = (cash - start_cash) / start_cash
                    daily_returns.append(ret)
                    
                    equity_curve.append(cash)
                    trade_dates.append(dates[i])
                
                if not equity_curve:
                    continue
                
                # Store data for plotting
                backtest_data[(market, model_name)] = {
                    'dates': trade_dates,
                    'prices': prices[test_start_idx+LOOKBACK : test_start_idx+LOOKBACK+len(trade_dates)], # Align with trade_dates
                    # Wait, trade_dates corresponds to i loop, which is range(test_start_idx, len(data)-1)
                    # prices[i] is the price at that step.
                    # So we can just store the subset of prices corresponding to the loop.
                    # But to be safe let's just use what we captured (though we didn't capture price series in a list, we have `prices` array)
                    # Let's simplify: record price alongside equity_curve?
                    # Actually trade_dates has dates. prices[i] was used.
                    'price_series': [prices[d_idx] for d_idx in range(test_start_idx, len(data)-1) if i >= LOOKBACK], # Logic is tricky to reconstruct indices
                    # Better: just capture them in the loop.
                }
                # Re-do capture in loop via replacement above? No, let's just append prices to a list in the loop.
                # OR just re-slice:
                # i ranges from test_start_idx to len(data)-2. 
                # if i < LOOKBACK continue.
                # So start_i = max(test_start_idx, LOOKBACK)
                start_i = max(test_start_idx, LOOKBACK)
                end_i = len(data) - 1
                # sliced_prices = prices[start_i:end_i] # This matches trade_dates length
                
                backtest_data[(market, regime_name, model_name)] = {
                    'dates': trade_dates,
                    'prices': [prices[si] for si in range(start_i, end_i)],
                    'events': trade_events
                }
    
                
                # --- Calculate Audit Metrics ---
                total_days = len(position_history)
                if total_days > 0:
                    pos_arr = np.array(position_history)
                    long_days = np.sum(pos_arr == 1)
                    short_days = np.sum(pos_arr == -1)
                    flat_days = np.sum(pos_arr == 0)
                    
                    exposure_long = (long_days / total_days) * 100
                    exposure_short = (short_days / total_days) * 100
                    
                    turnover = trade_count
                else:
                    exposure_long = 0.0
                    exposure_short = 0.0
                    turnover = 0
                
                # --- Calculate Metrics by Trend ---
                df_ret = pd.DataFrame({
                    'ret': daily_returns, 
                    'asset_ret': asset_returns,
                    'trend': trend_status
                })
    
                def calculate_metrics(series_ret, series_asset_ret, start_cap=INITIAL_CAPITAL):
                    if len(series_ret) == 0:
                        return 0.0, 0.0, 0.0, 0.0
                    
                    # compound returns
                    strat_cum = (np.prod(1 + series_ret) - 1) * 100
                    bnh_cum = (np.prod(1 + series_asset_ret) - 1) * 100
                    
                    # reconstruct a hypothetical equity curve for this subset
                    # Note: this is an approximation treating the disjoint days as continuous
                    subset_equity = start_cap * (1 + series_ret).cumprod()
                    
                    # Max DD
                    peak = subset_equity.cummax()
                    dd = (subset_equity - peak) / peak
                    max_dd = dd.min() * 100
                    
                    final_val = subset_equity.iloc[-1]
                    
                    return round(strat_cum, 2), round(bnh_cum, 2), round(max_dd, 2), round(final_val, 2)
                
                # 1. Overall
                strat_ret, bnh_ret, dd, final_val = calculate_metrics(df_ret['ret'], df_ret['asset_ret'])
                
                # Check for Exact Tie using "Official" Returns
                real_final_equity = equity_curve[-1]
                real_total_return = (real_final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                
                # B&H is geometric linking of asset returns over the same period
                # (Note: bnh_ret from helper is exactly this)
                
                tie_warning = False
                if abs(real_total_return - bnh_ret) < 1e-4: # Tolerance for float
                    tie_warning = True
                    print(f"  [WARNING] {model_name}: STRATEGY TIES B&H EXACTLY! (Ret: {real_total_return:.2f}%)")
                
                # Recalculate Max DD on the real continuous curve
                real_equity_series = pd.Series(equity_curve)
                real_max_dd = ((real_equity_series - real_equity_series.cummax()) / real_equity_series.cummax()).min() * 100
    
                results.append({
                    'Market': market,
                    'Trend': 'Overall',
                    'Regime': regime_name, # New column
                    'Model': model_name,
                    'Strategy Return (%)': round(real_total_return, 2),
                    'B&H Return (%)': round(bnh_ret, 2), # Using geometric linking of asset returns
                    'Max DD (%)': round(real_max_dd, 2),
                    'Final $': round(real_final_equity, 2),
                    'Long (%)': round(exposure_long, 1),
                    'Short (%)': round(exposure_short, 1),
                    'Turnover': turnover,
                    'Tie Warning': 'YES' if tie_warning else ''
                })
                
                print(f"    {model_name}: {real_total_return:+.2f}% (B&H: {bnh_ret:+.2f}%) | Long: {exposure_long:.0f}% | Trades: {turnover}")
    
                # 2. Uptrend Only
                up_data = df_ret[df_ret['trend'] == 'Uptrend']
                if len(up_data) > 0:
                    s_ret, b_ret, m_dd, f_val = calculate_metrics(up_data['ret'], up_data['asset_ret'])
                    results.append({
                        'Market': market,
                        'Trend': 'Uptrend',
                        'Model': model_name,
                        'Strategy Return (%)': s_ret,
                        'B&H Return (%)': b_ret,
                        'Max DD (%)': m_dd,
                        'Final $': f_val
                    })
    
                # 3. Downtrend Only
                down_data = df_ret[df_ret['trend'] == 'Downtrend']
                if len(down_data) > 0:
                    s_ret, b_ret, m_dd, f_val = calculate_metrics(down_data['ret'], down_data['asset_ret'])
                    results.append({
                        'Market': market,
                        'Trend': 'Downtrend',
                        'Model': model_name,
                        'Strategy Return (%)': s_ret,
                        'B&H Return (%)': b_ret,
                        'Max DD (%)': m_dd,
                        'Final $': f_val
                    })
    
    # Save results
    results_df = pd.DataFrame(results)
    # Reorder columns
    cols = ['Market', 'Regime', 'Trend', 'Model', 'Strategy Return (%)', 'B&H Return (%)', 'Max DD (%)', 
            'Final $', 'Long (%)', 'Short (%)', 'Turnover', 'Tie Warning']
    
    # Ensure all columns exist (for trend subsets that don't have audit cols)
    for c in cols:
        if c not in results_df.columns:
            results_df[c] = ''
            
    results_df = results_df[cols]
    
    results_df.to_csv("smart_backtest_results.csv", index=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY: BEST SMART MODEL PER MARKET (OVERALL)")
    print("=" * 70)
    
    # Filter only Overall for summary
    overall_df = results_df[results_df['Trend'] == 'Overall']
    
    for market in MARKETS:
        subset = overall_df[overall_df['Market'] == market]
        if len(subset) > 0:
            best = subset.loc[subset['Strategy Return (%)'].idxmax()]
            tie_msg = " [TIE!]" if best['Tie Warning'] == 'YES' else ""
            print(f"{market} (Regime: {best['Regime']}): {best['Model']} ({best['Strategy Return (%)']}% vs B&H {best['B&H Return (%)']}%) {tie_msg}")
            
            # Plot the best model
            best_model_name = best['Model']
            best_regime = best['Regime']
            
            if (market, best_regime, best_model_name) in backtest_data:
                bd = backtest_data[(market, best_regime, best_model_name)]
                plot_trades(f"{market}_{best_regime}", best_model_name, bd['dates'], bd['prices'], bd['events'], 
                            best['Strategy Return (%)'], best['B&H Return (%)'])
    
    print(f"\nFull results saved to smart_backtest_results.csv")

if __name__ == "__main__":
    run_smart_backtest()
