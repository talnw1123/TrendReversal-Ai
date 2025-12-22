import pandas as pd
import numpy as np
import ta
import warnings
from pandas.errors import PerformanceWarning

# Suppress warnings from ta library
warnings.filterwarnings('ignore', category=PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_features(df):
    """
    Calculates advanced features (Groups A-I) for the given DataFrame.
    Expects 'Close' and 'Volume' columns.
    Returns the DataFrame with added feature columns.
    """
    df = df.copy()
    
    # Parameters
    price_ma_periods = [5, 10, 20, 60, 120]
    vol_ma_periods = [5, 20, 60, 120]
    roc_periods = [1, 2, 4, 7, 12, 20, 33, 54, 68]
    epsilon = 1e-9 # To prevent division by zero

    # 0.1 Moving Averages
    for k in price_ma_periods:
        df[f'ma_{k}'] = df['Close'].rolling(window=k).mean()
        
    for k in vol_ma_periods:
        df[f'vma_{k}'] = df['Volume'].rolling(window=k).mean()

    # Group A: MA Uptrend (Binary)
    # A1) Price MA Uptrend
    for k in price_ma_periods:
        col = f'ma_{k}'
        df[f'p_ma_up_{k}'] = (df[col] > df[col].shift(1)).astype(int)
        
    # A2) Volume MA Uptrend
    for k in vol_ma_periods:
        col = f'vma_{k}'
        df[f'v_ma_up_{k}'] = (df[col] > df[col].shift(1)).astype(int)

    # Group B: MA Compare (Binary) & Group G: MA-MA Distance (Continuous)
    # Pairs: (5,10), (5,20), (5,60), (5,120), (10,20), (10,60), (10,120), (20,60), (20,120), (60,120)
    ma_pairs = [
        (5, 10), (5, 20), (5, 60), (5, 120),
        (10, 20), (10, 60), (10, 120),
        (20, 60), (20, 120),
        (60, 120)
    ]
    
    for s, l in ma_pairs:
        short_ma = df[f'ma_{s}']
        long_ma = df[f'ma_{l}']
        
        # B1) Price MA Compare
        df[f'p_ma_gt_{s}_{l}'] = (short_ma > long_ma).astype(int)
        
        # G) MA-MA Distance
        df[f'p_ma_dist_{s}_{l}'] = (short_ma - long_ma) / (long_ma + epsilon)

    # Group C: Price/Vol vs MA (Binary)
    # C1) Price vs Price MA
    for k in price_ma_periods:
        df[f'p_gt_ma_{k}'] = (df['Close'] > df[f'ma_{k}']).astype(int)
        
    # C2) Volume vs Volume MA
    for k in vol_ma_periods:
        df[f'v_gt_vma_{k}'] = (df['Volume'] > df[f'vma_{k}']).astype(int)

    # Group D: Price Disparity (Continuous)
    for k in price_ma_periods:
        ma = df[f'ma_{k}']
        df[f'p_disp_{k}'] = (df['Close'] - ma) / (ma + epsilon)

    # Group E: Price MA Gradient (Continuous)
    for k in price_ma_periods:
        ma = df[f'ma_{k}']
        prev_ma = ma.shift(1)
        df[f'p_ma_grad_{k}'] = (ma - prev_ma) / (prev_ma + epsilon)

    # Group F: Rate of Change (ROC) (Continuous)
    for d in roc_periods:
        prev_price = df['Close'].shift(d)
        df[f'roc_{d}'] = (df['Close'] - prev_price) / (prev_price + epsilon)

    # Group H: Trading Volume Disparity (Continuous)
    for k in vol_ma_periods:
        vma = df[f'vma_{k}']
        df[f'v_disp_{k}'] = (df['Volume'] - vma) / (vma + epsilon)

    # Group I: Volume MA Gradient (Continuous)
    for k in vol_ma_periods:
        vma = df[f'vma_{k}']
        prev_vma = vma.shift(1)
        df[f'v_ma_grad_{k}'] = (vma - prev_vma) / (prev_vma + epsilon)

    # ============================================
    # Group J: Technical Indicators (IMPROVED)
    # ============================================
    
    close = df['Close']
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    
    # J1) RSI (Relative Strength Index)
    for period in [14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + epsilon)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        # Normalized RSI (0 to 1)
        df[f'rsi_{period}_norm'] = df[f'rsi_{period}'] / 100
    
    # J2) MACD (Moving Average Convergence Divergence)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = (ema_12 - ema_26) / (close + epsilon)  # Normalized
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # J3) Bollinger Bands
    for period in [20]:
        bb_mid = close.rolling(window=period).mean()
        bb_std = close.rolling(window=period).std()
        df[f'bb_upper_{period}'] = bb_mid + (2 * bb_std)
        df[f'bb_lower_{period}'] = bb_mid - (2 * bb_std)
        # BB Position (where is price relative to bands, -1 to 1)
        df[f'bb_pos_{period}'] = (close - bb_mid) / (2 * bb_std + epsilon)
        # BB Width (volatility indicator)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / (bb_mid + epsilon)
    
    # J4) ATR (Average True Range) - Volatility
    for period in [14]:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(window=period).mean() / (close + epsilon)  # Normalized
    
    # J5) Stochastic Oscillator
    for period in [14]:
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        df[f'stoch_k_{period}'] = (close - lowest_low) / (highest_high - lowest_low + epsilon)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
    
    # J6) Momentum
    for period in [10, 20]:
        df[f'momentum_{period}'] = (close - close.shift(period)) / (close.shift(period) + epsilon)
    
    # J7) Williams %R
    for period in [14]:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        df[f'williams_r_{period}'] = (highest_high - close) / (highest_high - lowest_low + epsilon)
    
    # J8) Price Position in Range
    for period in [20, 60]:
        highest = high.rolling(window=period).max()
        lowest = low.rolling(window=period).min()
        df[f'price_pos_{period}'] = (close - lowest) / (highest - lowest + epsilon)

    # ============================================
    # Group K: TA Library Features
    # ============================================
    try:
        # Ensure Open exists (fallback to Close if missing)
        if 'Open' not in df.columns:
            df['Open'] = df['Close']
            
        # Add all features from ta library
        df = ta.add_all_ta_features(
            df, 
            open="Open", 
            high="High", 
            low="Low", 
            close="Close", 
            volume="Volume", 
            fillna=True
        )
        # Defragment the dataframe
        df = df.copy()
    except Exception as e:
        print(f"Warning: Could not add TA library features: {e}")

    # Clean up: Fill NaNs instead of dropping to preserve data for shorter datasets
    # Use forward fill first, then backward fill for remaining NaNs
    df = df.ffill().bfill()
    
    return df

def get_feature_columns(df):
    """
    Returns the list of selected features.
    Prioritizes 'model/selected_features.json' if it exists.
    Otherwise returns hardcoded default features.
    """
    import json
    import os
    
    # Try to load selected features from JSON
    # Check possible paths (running from root or model/)
    json_paths = ["model/selected_features.json", "selected_features.json", "../model/selected_features.json"]
    
    selected_features_from_file = []
    
    for path in json_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    selected_features_from_file = json.load(f)
                if selected_features_from_file:
                    # Validate that these features exist in the current df
                    # Some features might be dynamic (like from ta lib) and might not be in df if calculation failed
                    # But usually they should be there.
                    valid_loaded = [f for f in selected_features_from_file if f in df.columns]
                    
                    if len(valid_loaded) > 0:
                        # If we have valid features, return them
                        if len(valid_loaded) < len(selected_features_from_file):
                            print(f"Warning: {len(selected_features_from_file) - len(valid_loaded)} selected features were not found in DataFrame.")
                        
                        return valid_loaded
            except Exception as e:
                print(f"Warning: Could not load selected features from {path}: {e}")
            break # Stop after finding the first existing file

    # Fallback to hardcoded defaults (The Top 88 Features from analysis)
    default_selected = [
        "atr_14", "p_ma_grad_120", "volume_obv", "volume_adi", "volume_nvi",
        "p_ma_dist_60_120", "volume_vpt", "others_cr", "volatility_dcw", "volatility_bbw",
        "volatility_kcw", "bb_width_20", "p_ma_dist_20_120", "volume_cmf", "trend_mass_index",
        "volatility_atr", "trend_adx", "p_ma_dist_10_120", "momentum_pvo_signal", "volume_mfi",
        "roc_68", "trend_visual_ichimoku_b", "p_ma_dist_20_60", "p_disp_120", "momentum_pvo",
        "volume_sma_em", "trend_macd_signal", "momentum_pvo_hist", "volume_fi", "volatility_ui",
        "v_ma_grad_120", "trend_stc", "v_disp_120", "trend_adx_neg", "v_ma_grad_20",
        "momentum_ppo_signal", "p_ma_dist_10_60", "p_ma_dist_5_120", "trend_adx_pos", "p_ma_grad_60",
        "momentum_ao", "trend_kst_diff", "trend_macd", "roc_54", "p_ma_dist_5_60",
        "trend_macd_diff", "momentum_uo", "trend_vortex_ind_pos", "trend_kst_sig", "momentum_stoch_rsi_d",
        "v_ma_grad_60", "v_disp_60", "momentum_stoch_signal", "stoch_d_14", "macd_signal",
        "trend_aroon_ind", "v_disp_20", "p_ma_dist_10_20", "trend_vortex_ind_neg", "trend_cci",
        "rsi_14_norm", "bb_pos_20", "volatility_kcp", "trend_dpo", "trend_vortex_ind_diff",
        "rsi_21_norm", "price_pos_60", "trend_trix", "momentum_tsi", "p_ma_dist_5_20",
        "momentum_ppo_hist", "v_disp_5", "macd_hist", "trend_kst", "roc_33",
        "volatility_dcp", "v_ma_grad_5", "price_pos_20", "volatility_bbp", "momentum_wr",
        "volatility_dcl", "momentum_stoch_rsi_k", "p_ma_grad_5", "p_ma_dist_5_10", "p_disp_5",
        "momentum_ppo", "volatility_bbl", "momentum_rsi"
    ]
    
    selected = default_selected
    
    # Verify they exist in df
    valid_selected = [f for f in selected if f in df.columns]
    return valid_selected

def get_selected_features(df):
    """
    Alias for get_feature_columns to maintain compatibility with training script.
    """
    return get_feature_columns(df)
