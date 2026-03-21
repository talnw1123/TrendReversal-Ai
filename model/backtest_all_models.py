"""
Backtest using the BEST Specialized Model for each Market & Trend.
Reads champions from: separate_models_comparison.csv
Loads models from: model_{Market}_{Trend}/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import joblib
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import warnings
from pandas.errors import PerformanceWarning
import ta

# Suppress warnings
warnings.filterwarnings('ignore', category=PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

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

def get_selected_features(df):
    """
    Returns the list of selected features.
    Prioritizes 'model/selected_features.json' if it exists.
    Otherwise returns hardcoded default features.
    """
    import json
    
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
                    valid_loaded = [f for f in selected_features_from_file if f in df.columns]
                    if len(valid_loaded) > 0:
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

# Try importing from feature_importance, else define dummy
# The instruction implies using the local calculate_features for add_technical_features
def add_technical_features(df):
    """Placeholder for add_technical_features, now using local calculate_features."""
    return calculate_features(df)

# Parameters
LOOKBACK = 30 # Matches training lookback
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001
CONFIDENCE_THRESHOLD = 0.65

# MOO-Optimized Parameters (per market)
OPTIMIZED_PARAMS = {
    'Thai': {'confidence': 0.5132, 'long': 0.5413, 'short': 0.3166, 'stop_loss': 0.0793},
    'UK': {'confidence': 0.5169, 'long': 0.5232, 'short': 0.3251, 'stop_loss': 0.1105},
    'Gold': {'confidence': 0.5392, 'long': 0.5488, 'short': 0.3533, 'stop_loss': 0.0320},
    'US': {'confidence': 0.5105, 'long': 0.5393, 'short': 0.3019, 'stop_loss': 0.1496},
    # 'BTC': {'confidence': 0.5955, 'long': 0.5765, 'short': 0.3294, 'stop_loss': 0.0596}, # Reverted to Original Default
    'BTC': {'confidence': 0.55, 'long': 0.54, 'short': 0.46, 'stop_loss': 0.20}, # Original Default Logic (Loose SL)
}

# Market-Specific Strategy Configuration
# 'active': Neutral -> Cash, No Trailing Stop (or specific setup)
# 'smart_hold': Neutral -> Hold, Use Trailing Stop
# 'defensive': Tight Stop, Neutral -> Cash
MARKET_STRATEGIES = {
    'US': 'smart_hold',
    'UK': 'smart_hold', # Try wider trailing stop
    'Gold': 'smart_hold',
    'BTC': 'active',
    'Thai': 'active' 
}

# Market-Specific Trailing Stops (only used if strategy is 'smart_hold')
MARKET_TRAILING_STOPS = {
    'US': 0.15,    # 15% Trailing (Beat B&H)
    'UK': 0.20,    # 20% Trailing (Match B&H closely)
    'Gold': 0.15,  # 15% Trailing (Lock profits)
    'BTC': 0.0,    # Active Trading (No Trailing)
    'Thai': 0.05   # Active/Tight Stop
}

# Market-Specific Long Only Configuration
MARKET_LONG_ONLY = {
    'US': True,    # Stocks usually go up
    'UK': True,    # Stocks usually go up
    'Gold': True,  # Commodities trend well
    'BTC': False,  # Two-way market
    'Thai': False  # Two-way market
}

MARKETS = ['US', 'UK', 'Thai', 'Gold', 'BTC']
TRENDS = ['uptrend', 'downtrend']

# Model directories (Updated to use new paths)
MODEL_BASE_DIR = 'all_trad/separate_models'
REGIME_MODEL_DIR = 'all_trad/regime_models'

# Import RegimeDetector for regime-based model selection
import sys
sys.path.insert(0, 'model')
from regime_detection import RegimeDetector

def get_ticker(market):
    tickers = {
        'US': '^GSPC',
        'UK': '^FTSE',
        'Thai': '^SET.BK',
        'Gold': 'GC=F',
        'BTC': 'BTC-USD'
    }
    return tickers.get(market)

def load_model_and_objects(model_dir, model_name):
    """
    Load model, scaler, and label_encoder.
    Returns: model, scaler, le, model_type ('keras' or 'ml')
    """
    model = None
    scaler = None
    le = None
    model_type = 'ml'
    
    # Load Model
    # Try keras first
    p_keras = os.path.join(model_dir, f"{model_name}.keras")
    p_pkl = os.path.join(model_dir, f"{model_name}.pkl")
    
    if os.path.exists(p_keras):
        model = load_model(p_keras)
        model_type = 'keras'
    elif os.path.exists(p_pkl):
        model = joblib.load(p_pkl)
        model_type = 'ml'
        
    # Load Objects
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        
    return model, scaler, le, model_type

def calculate_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min() * 100

def plot_backtest(dates, equity_curve, prices, signals, market, trend, model_name, final_return, bnh_return, regime=None):
    """Generate and save backtest plot with Entry/Exit markers and Regime Background."""
    save_dir = "model/backtest_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Align data
    # dates, prices, signals are all length N
    # equity_curve length is N-1 (starts from step 1)
    
    plot_dates = dates[1:]
    plot_prices = prices[1:]
    plot_bnh = (prices[1:] / prices[0]) * INITIAL_CAPITAL
    plot_equity = equity_curve
    plot_signals = signals[:-1] # shift to align with decision points
    
    # Create Trade Markers
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    # 1 = Buy/Hold Long, 0 = Short (if active), 99 = Neutral/Cash
    # Detecting CHANGES in position
    prev_pos = 0 # Assuming start neutral
    
    for i in range(len(plot_dates)):
        sig = plot_signals[i]
        curr_pos = 0
        
        if sig == 1: curr_pos = 1
        elif sig == 0: curr_pos = -1 # Assuming separate model uses -1 for bear
        elif sig == 99: curr_pos = 0
        
        if curr_pos != prev_pos:
            if curr_pos == 1: # Entry Long
                buy_dates.append(plot_dates[i])
                buy_prices.append(plot_prices[i])
            elif curr_pos == -1: # Entry Short (if supported) or Exit Long
                 # For now separate models are Directional. 
                 # If we treat 0 as Short Entry:
                 sell_dates.append(plot_dates[i])
                 sell_prices.append(plot_prices[i])
            elif curr_pos == 0: # Exit to Cash
                 # Mark as exit
                 sell_dates.append(plot_dates[i])
                 sell_prices.append(plot_prices[i])
        
        prev_pos = curr_pos

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # 1. Price Chart with Markers
    ax1.plot(plot_dates, plot_prices, label='Price', color='black', alpha=0.6)
    ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy/Long', zorder=5)
    ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell/Exit', zorder=5)
    ax1.set_title(f"{market} {trend} - Price & Signals")
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Regime Background
    if regime is not None and len(regime) == len(dates):
        # Align regime with plot_dates (dates[1:])
        plot_regime = regime[1:]
        
        # Find segments
        start_idx = 0
        current_val = plot_regime[0]
        
        for i in range(1, len(plot_regime)):
            if plot_regime[i] != current_val:
                # End segment
                color = 'green' if current_val == 1 else 'red'
                # Use dates to span
                ax1.axvspan(plot_dates[start_idx], plot_dates[i], color=color, alpha=0.1)
                ax2.axvspan(plot_dates[start_idx], plot_dates[i], color=color, alpha=0.1)
                
                start_idx = i
                current_val = plot_regime[i]
                
        # Last segment
        color = 'green' if current_val == 1 else 'red'
        ax1.axvspan(plot_dates[start_idx], plot_dates[-1], color=color, alpha=0.1)
        ax2.axvspan(plot_dates[start_idx], plot_dates[-1], color=color, alpha=0.1)

    
    # 2. Equity Curve
    ax2.plot(plot_dates, plot_bnh, label=f'Buy & Hold ({bnh_return:.2f}%)', color='gray', linestyle='--')
    ax2.plot(plot_dates, plot_equity, label=f'Strategy ({final_return:.2f}%)', color='blue', linewidth=2)
    ax2.set_title(f"Equity Curve ({model_name})")
    ax2.set_ylabel('Equity ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_dir}/{market}_{trend}_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    print(f"  Plot saved to {filename}")

def load_market_test_data(market, trend):
    """Load and prepare test data for a market/trend with warmup."""
    from features import calculate_features, get_selected_features
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path relative to script: ../trend_data_manual/split
    base_split_dir = os.path.join(script_dir, "../trend_data_manual/split")
    
    # Search specifically for this market and trend test file
    test_pattern = f"*{market}*{trend}*_labeled.csv"
    val_pattern = f"*{market}*{trend}*_labeled.csv"
    
    test_files = glob.glob(os.path.join(base_split_dir, f"test/**/{test_pattern}"), recursive=True)
    
    if not test_files:
        # Try finding in root of test split
        test_files = glob.glob(os.path.join(base_split_dir, f"test/{test_pattern}"))
        
    if not test_files:
        return None, None, None, None
    
    try:
        df_test = pd.read_csv(test_files[0], index_col=0, parse_dates=True).sort_index()
    except:
        return None, None, None, None
        
    test_start_date = df_test.index.min()
    
    # Warmup with validation data (to fill lookback window)
    val_files = glob.glob(os.path.join(base_split_dir, f"val/**/{val_pattern}"), recursive=True)
    if not val_files:
         val_files = glob.glob(os.path.join(base_split_dir, f"val/{val_pattern}"))
         
    if val_files:
        try:
            df_val = pd.read_csv(val_files[0], index_col=0, parse_dates=True).sort_index()
            df_combined = pd.concat([df_val, df_test])
            df_combined = df_combined[~df_combined.index.duplicated(keep='first')].sort_index()
        except:
             df_combined = df_test
    else:
        df_combined = df_test
    
    df_combined = calculate_features(df_combined)
    if df_combined.empty:
        return None, None, None, None
    
    feature_cols = get_selected_features(df_combined)
    data = df_combined[feature_cols].values
    prices = df_combined['Close'].values
    dates = df_combined.index
    
    # Extract valid test windows
    X_list = []
    valid_indices = []
    
    for i in range(LOOKBACK, len(data)):
        # Only keep windows where the TARGET index is in the test set
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
        
        if signal == 1: # Bullish
             target_pos = 1
        elif signal == 0: # Bearish
             target_pos = -1
        elif signal == 99: # Neutral (Low Confidence)
             target_pos = 0
             
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

def run_simulation_moo(signals, prices, confidences=None, regimes=None, stop_loss_pct=0.05, trailing_stop_pct=0.10, long_only=False, strategy_mode='active', initial_capital=10000.0, transaction_cost=0.001, leverage=1.0, sizing_mode='conservative', min_position=0.0):
    """
    Run trading simulation with MOO-optimized stop-loss, trailing stop, continuous Position Sizing, and optional Leverage.
    
    sizing_mode: 'conservative', 'moderate', or 'aggressive' — controls confidence-to-size mapping
    min_position: minimum position to hold when in uptrend regime (0.0 = no floor)
    regimes: array of 0/1 indicating downtrend/uptrend per day (for min_position logic)
    """
    # Sizing tier lookup tables
    SIZING_TIERS = {
        'conservative': {70: 1.0, 60: 0.6,  50: 0.3, 0: 0.1},
        'moderate':     {70: 1.0, 60: 0.8,  50: 0.5, 0: 0.3},
        'aggressive':   {70: 1.0, 60: 0.9,  50: 0.7, 0: 0.5},
    }
    tiers = SIZING_TIERS.get(sizing_mode, SIZING_TIERS['conservative'])
    
    cash = initial_capital
    position = 0.0  # Float position (-1.0 to 1.0)
    entry_price = 0.0
    highest_price = 0.0 # For trailing stop
    equity_curve = []
    trade_points = []  # List of (index, 'buy'|'sell') for plotting
    n_trades = 0
    wins = 0
    cooldown = 0
    
    for i in range(len(signals) - 1):
        signal = signals[i]
        price = prices[i]
        next_price = prices[i+1]
        
        # Determine sizing multiplier based on confidence + sizing_mode
        size_multiplier = 1.0
        if confidences is not None:
            conf = confidences[i]
            if conf >= 0.70:
                size_multiplier = tiers[70] * leverage  # Leverage on high confidence
            elif conf >= 0.60:
                size_multiplier = tiers[60] * leverage  # Leverage also at medium-high confidence
            elif conf >= 0.50:
                size_multiplier = tiers[50]
            else:
                size_multiplier = tiers[0]
        
        # Track highest/lowest price for trailing stop
        if position > 0:
            if price > highest_price:
                highest_price = price
        elif position < 0:
            if entry_price > 0 and (highest_price == 0 or price < highest_price):
                highest_price = price
        
        # Decrement cooldown
        if cooldown > 0:
            cooldown -= 1
        
        # Check Exits (Stop Loss & Trailing Stop) first
        if position != 0 and entry_price > 0:
            if position > 0:
                pnl_pct = (price - entry_price) / entry_price
                if trailing_stop_pct > 0:
                    drawdown_from_peak = (highest_price - price) / highest_price
                    if drawdown_from_peak > trailing_stop_pct:
                        cost = cash * abs(position) * transaction_cost
                        cash -= cost
                        if pnl_pct > 0: wins += 1
                        n_trades += 1
                        trade_points.append((i, 'sell'))
                        position = 0.0
                        entry_price = 0.0
                        highest_price = 0.0
                        equity_curve.append(cash)
                        continue
            else:
                 pnl_pct = (entry_price - price) / entry_price
                 if trailing_stop_pct > 0 and highest_price > 0:
                     drawup_from_bottom = (price - highest_price) / highest_price
                     if drawup_from_bottom > trailing_stop_pct:
                         cost = cash * abs(position) * transaction_cost
                         cash -= cost
                         if pnl_pct > 0: wins += 1
                         n_trades += 1
                         trade_points.append((i, 'sell'))
                         position = 0.0
                         entry_price = 0.0
                         highest_price = 0.0
                         equity_curve.append(cash)
                         continue
            
            # Stop Loss Check
            if pnl_pct < -stop_loss_pct:
                cost = cash * abs(position) * transaction_cost
                cash -= cost
                if pnl_pct > 0: wins += 1
                n_trades += 1
                trade_points.append((i, 'sell'))
                position = 0.0
                entry_price = 0.0
                highest_price = 0.0
                cooldown = 3 
                equity_curve.append(cash)
                continue
        
        # Determine target position based on signal
        target_pos = 0.0 
        
        if cooldown == 0:
            if signal == 1:
                target_pos = 1.0 * size_multiplier
            elif signal == 0:
                target_pos = 0.0 if long_only else -1.0 * size_multiplier
            elif signal == 99:
                if strategy_mode == 'smart_hold':
                    target_pos = position
                else:
                    target_pos = 0.0
            
            # Regime-aware minimum position floor
            # In uptrend, always hold at least min_position (prevents full cash exit)
            if min_position > 0 and regimes is not None and i < len(regimes):
                if regimes[i] == 1 and target_pos >= 0:  # Uptrend regime
                    target_pos = max(target_pos, min_position)
                elif regimes[i] == 0 and not long_only and target_pos <= 0:  # Downtrend regime
                    target_pos = min(target_pos, -min_position)
        else:
            target_pos = 0.0
            
        # Execute Trade (Adjust position)
        if position != target_pos:
            cost = cash * abs(target_pos - position) * transaction_cost
            cash -= cost
            
            # If closing or flipping a position, record trade stats for the OLD position
            if position != 0.0 and entry_price > 0:
                # We are closing at least part of the position, or flipping it completely
                if position > 0:
                    trade_pnl = (price - entry_price) / entry_price
                else:
                    trade_pnl = (entry_price - price) / entry_price
                
                # Only count as a trade if we are exiting to neutral, or flipping direction
                if target_pos == 0.0 or (position > 0 and target_pos < 0) or (position < 0 and target_pos > 0):
                    if trade_pnl > 0: wins += 1
                    n_trades += 1
            
            # Record trade point for plotting
            if position <= 0 and target_pos > 0:
                trade_points.append((i, 'buy'))
            elif position >= 0 and target_pos < 0:
                trade_points.append((i, 'sell'))
            elif target_pos == 0.0 and position != 0.0:
                trade_points.append((i, 'sell'))
            
            # If entering a new position direction, reset entry price
            if (position <= 0 and target_pos > 0) or (position >= 0 and target_pos < 0):
                entry_price = price
                highest_price = price
                
            position = target_pos
            
        # Daily Equity Update based on continuous position (-1.0 to 1.0)
        cash *= (1 + position * (next_price - price) / price)
        
        equity_curve.append(cash)
    
    win_rate = (wins / n_trades * 100) if n_trades > 0 else 0.0
    return equity_curve, n_trades, win_rate, trade_points

def load_best_models_map():
    """
    Load best prediction models from separate_models_comparison.csv.
    Selects the model with highest Accuracy for each Market/Trend combination.
    """
    best_map = {}
    csv_path = os.path.join(MODEL_BASE_DIR, 'separate_models_comparison.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}. Falling back to directory scan.")
        # Fallback to directory scan
        return _scan_models_fallback()
    
    try:
        df = pd.read_csv(csv_path)
        
        # For each Market/Trend, get the row with highest Accuracy
        for (market, trend), group in df.groupby(['Market', 'Trend']):
            best_row = group.loc[group['Accuracy'].idxmax()]
            best_model = best_row['Model']
            best_acc = best_row['Accuracy']
            
            # Verify model file exists
            model_folder = os.path.join(MODEL_BASE_DIR, f"model_{market}_{trend}")
            p_keras = os.path.join(model_folder, f"{best_model}.keras")
            p_pkl = os.path.join(model_folder, f"{best_model}.pkl")
            
            if os.path.exists(p_keras) or os.path.exists(p_pkl):
                best_map[(market, trend)] = best_model
                print(f"  Best for {market}/{trend}: {best_model} (Acc: {best_acc:.2f}%)")
            else:
                print(f"  Warning: Model {best_model} not found for {market}/{trend}")
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return _scan_models_fallback()
    
    return best_map

def _scan_models_fallback():
    """Fallback: Scan directory for models if CSV is not available."""
    best_map = {}
    model_priority = ['RandomForest', 'XGBoost', 'LSTM', 'CNN', 'MLP', 'SVM', 'Transformer']
    
    subdirs = glob.glob(os.path.join(MODEL_BASE_DIR, "model_*_*"))
    
    for folder in subdirs:
        basename = os.path.basename(folder)
        parts = basename.split('_')
        if len(parts) < 3: continue
        
        market = parts[1]
        trend = parts[2]
        
        for m_name in model_priority:
            p1 = os.path.join(folder, f"{m_name}.keras")
            p2 = os.path.join(folder, f"{m_name}.pkl")
            
            if os.path.exists(p1) or os.path.exists(p2):
                best_map[(market, trend)] = m_name
                break
                
    return best_map

def quick_backtest_model(market, trend, model_name, threshold_configs=None):
    """
    Quick backtest a single model for a market/trend combination.
    Tests multiple threshold configurations and returns the best metrics.
    Returns dict with: return, max_dd, sharpe, win_rate, best_config (or None on failure).
    """
    try:
        # 1. Load data
        ticker = get_ticker(market)
        if not ticker:
            return None
        
        df = yf.download(ticker, period="5y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty or len(df) < 200:
            return None
        
        df = add_technical_features(df)
        df = df.dropna()
        
        if len(df) < LOOKBACK + 50:
            return None
        
        feature_cols = get_selected_features(df)
        data = df[feature_cols].values
        prices = df['Close'].values
        
        # 2. Load model
        model_dir = os.path.join(MODEL_BASE_DIR, f"model_{market}_{trend}")
        model, scaler, le, model_type = load_model_and_objects(model_dir, model_name)
        
        if model is None:
            return None
        
        # 3. Batch predict all probabilities first (speed optimization)
        sequences = []
        for i in range(LOOKBACK, len(data)):
            sequence = data[i-LOOKBACK:i].copy()
            mean = np.nanmean(sequence, axis=0)
            std = np.nanstd(sequence, axis=0) + 1e-9
            sequence = (sequence - mean) / std
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
            sequences.append(sequence)
        
        if len(sequences) < 10:
            return None
        
        X_batch = np.array(sequences)  # (N, LOOKBACK, features)
        
        try:
            if model_type == 'ml':
                X_flat = X_batch.reshape(len(X_batch), -1)
                probs = model.predict_proba(X_flat)
            else:
                probs = model.predict(X_batch, verbose=0, batch_size=512)
            
            bull_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        except Exception as e:
            return None
        
        sim_prices = prices[LOOKBACK:]
        strategy_mode = MARKET_STRATEGIES.get(market, 'active')
        trailing_stop = MARKET_TRAILING_STOPS.get(market, 0.0)
        long_only = MARKET_LONG_ONLY.get(market, False)
        
        # 4. Default threshold configs to test (grid search for better results)
        if threshold_configs is None:
            moo_params = OPTIMIZED_PARAMS.get(market, {
                'confidence': 0.55, 'long': 0.54, 'short': 0.46, 'stop_loss': 0.05
            })
            base_conf = moo_params['confidence']
            base_long = moo_params['long']
            base_short = moo_params['short']
            base_sl = moo_params['stop_loss']
            
            threshold_configs = [
                # Original MOO params
                {'confidence': base_conf, 'long': base_long, 'short': base_short, 'stop_loss': base_sl},
                # Looser thresholds (more trades)
                {'confidence': max(0.50, base_conf - 0.03), 'long': max(0.51, base_long - 0.03), 
                 'short': min(0.49, base_short + 0.03), 'stop_loss': base_sl},
                # Tighter thresholds (fewer, higher-quality trades)
                {'confidence': min(0.70, base_conf + 0.05), 'long': min(0.70, base_long + 0.05), 
                 'short': max(0.30, base_short - 0.05), 'stop_loss': base_sl},
                # Lower stop loss
                {'confidence': base_conf, 'long': base_long, 'short': base_short, 
                 'stop_loss': max(0.02, base_sl * 0.5)},
                # Higher stop loss 
                {'confidence': base_conf, 'long': base_long, 'short': base_short, 
                 'stop_loss': min(0.20, base_sl * 2.0)},
                # Very loose (try to trade more)
                {'confidence': 0.50, 'long': 0.52, 'short': 0.48, 'stop_loss': 0.10},
            ]
        
        # 5. Test each config and collect all results
        all_results = []
        for cfg in threshold_configs:
            conf_thresh = cfg['confidence']
            long_thresh = cfg['long']
            short_thresh = cfg['short']
            stop_loss = cfg['stop_loss']
            
            # Generate signals with this config
            signals = []
            for bp in bull_probs:
                confidence = max(bp, 1 - bp)
                if confidence >= conf_thresh:
                    if bp >= long_thresh:
                        signals.append(1)
                    elif bp <= short_thresh:
                        signals.append(0)
                    else:
                        signals.append(99)
                else:
                    signals.append(99)
            
            # Run simulation
            equity_curve, n_trades, win_rate, _ = run_simulation_moo(
                signals, sim_prices,
                stop_loss_pct=stop_loss,
                trailing_stop_pct=trailing_stop,
                long_only=long_only,
                strategy_mode=strategy_mode
            )
            
            if not equity_curve or len(equity_curve) == 0:
                continue
            
            # Calculate metrics
            equity = np.array(equity_curve)
            total_return = ((equity[-1] / INITIAL_CAPITAL) - 1) * 100
            
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            max_dd = np.min(drawdown) * 100
            
            daily_returns = np.diff(equity) / equity[:-1]
            if len(daily_returns) > 0 and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe = 0.0
            
            all_results.append({
                'return': round(total_return, 2),
                'max_dd': round(max_dd, 2),
                'sharpe': round(sharpe, 2),
                'win_rate': round(win_rate, 2),
                'trades': n_trades,
                'config': cfg,
            })
        
        if not all_results:
            return None
        
        # Return the best result (by return as primary, sharpe as secondary)
        best = max(all_results, key=lambda r: (r['sharpe'], r['return']))
        return best
        
    except Exception as e:
        print(f"    Quick backtest error for {model_name}: {e}")
        return None


def load_best_models_map_moo(preference='balanced'):
    """
    Load best models directly from the generated MOO selection file.
    Requested by user to strictly use: all_trad/separate_models/moo_model_selection.csv
    """
    best_map = {}
    
    csv_path = os.path.join(MODEL_BASE_DIR, 'moo_model_selection.csv')
    if os.path.exists(csv_path):
        print(f"  Loading MOO models from CSV: {csv_path}")
        try:
            cache_df = pd.read_csv(csv_path)
            for _, row in cache_df.iterrows():
                market = row['Market']
                trend = row['Trend']
                model_name = row['Best_Model']
                
                # We assume the model exists since it was selected
                best_map[(market, trend)] = model_name
                print(f"  Best for {market}/{trend}: {model_name}")
                
            return best_map
        except Exception as e:
            print(f"  Error reading MOO CSV: {e}")
            
    print(f"Warning: {csv_path} not found. Returning empty map.")
    return best_map


def load_best_regime_methods():
    """
    Load best regime detection method from regime_evaluation_summary.csv.
    Selects the method where Is_Best == True for each Market.
    Returns: dict {market: method_name}
    """
    best_regime = {}
    csv_path = 'all_trad/regime_models/regime_evaluation_summary.csv'
    
    if not os.path.exists(csv_path):
        print(f"Warning: Regime CSV not found at {csv_path}. Using default HMM.")
        for market in MARKETS:
            best_regime[market] = 'HMM'
        return best_regime
    
    try:
        df = pd.read_csv(csv_path)
        
        # Filter for the best models
        best_df = df[df['Is_Best'] == True]
        
        for _, row in best_df.iterrows():
            market = row['Market']
            best_method = row['Method']
            
            best_regime[market] = best_method
            print(f"  Best Regime for {market}: {best_method}")
            
        # Ensure all MARKETS have a fallback if missing
        for market in MARKETS:
            if market not in best_regime:
                best_regime[market] = 'HMM'
                
    except Exception as e:
        print(f"Error reading regime CSV: {e}")
        for market in MARKETS:
            best_regime[market] = 'HMM'
    
    return best_regime

def load_best_thresholds():
    """
    Load best thresholds from backtest_champion_results.csv.
    Returns: dict {(market, trend): threshold}
    """
    thresholds = {}
    csv_path = 'backtest_champion_results.csv'
    
    # Default thresholds if CSV not found
    default_thresholds = {
        ('US', 'uptrend'): 0.53, ('US', 'downtrend'): 0.51,
        ('UK', 'uptrend'): 0.85, ('UK', 'downtrend'): 0.50,
        ('Thai', 'uptrend'): 0.50, ('Thai', 'downtrend'): 0.50,
        ('Gold', 'uptrend'): 0.50, ('Gold', 'downtrend'): 0.65,
        ('BTC', 'uptrend'): 0.52, ('BTC', 'downtrend'): 0.52,
    }
    
    if not os.path.exists(csv_path):
        print(f"  Using default thresholds (CSV not found)")
        return default_thresholds
    
    try:
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            market = row['Market']
            trend = row['Trend']
            thresh = row['Best Threshold']
            thresholds[(market, trend)] = thresh
            
        print(f"  Loaded {len(thresholds)} thresholds from {csv_path}")
                
    except Exception as e:
        print(f"Error reading thresholds CSV: {e}")
        return default_thresholds
    
    return thresholds

def main():
    print("=" * 70)
    print("BACKTEST: BEST SPECIALIZED MODELS (MOO Selection)")
    print("=" * 70)
    
    # Load best prediction models using MOO composite score
    print("\n[Prediction Models - MOO Multi-Objective Selection]")
    best_models_map = load_best_models_map_moo()
    if not best_models_map:
        print("No champion models found. Please run training first.")
        return

    print(f"\n-> Loaded {len(best_models_map)} best prediction models.")
    
    # Load best regime detection methods from CSV
    print("\n[Regime Detection - from regime_comparison_results.csv]")
    best_regime_map = load_best_regime_methods()
    print(f"\n-> Loaded {len(best_regime_map)} best regime methods.")
    
    results = []
    
    for market in MARKETS:
        regime_method = best_regime_map.get(market, 'HMM')
        print(f"\n>> Simulating Market: {market} (Regime: {regime_method})")
        
        # Determine Market Regime (Uptrend/Downtrend)
        # For simplicity in this backtest, we run BOTH trend models on the entire dataset 
        # but realistically we should switch based on regime detection.
        # Here we will just backtest each separately to see performance in their specific conditions
        # OR we can combine them? User asked to "Backtest All Models".
        # Let's run both Uptown and Downtrend models for this market and see which performs better overall
        # or just report both.
        
        for trend in TRENDS:
            if (market, trend) not in best_models_map:
                print(f"  No model found for {market} ({trend}). Skipping.")
                continue
                
            model_name = best_models_map[(market, trend)]
            print(f"  Using Model: {model_name} for {market} {trend}")
            
            # 1. Load Data
            # Note: Tickers are in config.py, let's assume we can fetch or load
            ticker = get_ticker(market) # Helper needed or import
            if not ticker: continue
            
            print(f"  Loading data for {ticker}...")
            try:
                # Load data (reuses logic from train_separate_models or similar)
                # We need fresh data for backtesting (entire history?)
                # For consistency, let's download 2y data
                df = yf.download(ticker, period="2y", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df.empty:
                    print("    Empty data. Skipping.")
                    continue
                    
                # Preprocess
                # We need to apply the SAME technical indicators as training
                # Import ad_technical_features from feature_importance.py or defined locally?
                # It is likely defined in features.py or similar. 
                # Let's check imports.
                # Assuming feature_importance has 'add_technical_features'
                
                df = add_technical_features(df)
                df = df.dropna()
                
            except Exception as e:
                print(f"    Error loading data: {e}")
                continue
            
            # 2. Load Model
            # Model path: MODEL_BASE_DIR / model_{market}_{trend} / {model_name}.[keras|pkl]
            model_dir = os.path.join(MODEL_BASE_DIR, f"model_{market}_{trend}")
            try:
                model, scaler, label_encoder, model_type = load_model_and_objects(model_dir, model_name)
            except Exception as e:
                print(f"    Error loading model: {e}")
                continue
                
            if not model:
                print(f"    Model object is None. Skipping.")
                continue
                
            # 3. Prepare Features
            # Drop non-numeric columns (like Date if present) and target if present
            # Assuming add_technical_features added what we need
            
            # Identify feature columns
            # Determine model type and features expected
            # 3. Prepare Features
            
            # Identify feature columns
            selected_cols = get_selected_features(df)
            
            if not selected_cols:
                 print("    Warning: No selected features found.")
                 continue
                 
            # Check overlap
            missing = [f for f in selected_cols if f not in df.columns]
            if missing:
                for m in missing:
                    df[m] = 0
            
            feature_data = df[selected_cols].values
            
            # Align LOOKBACK
            # Training script likely used LOOKBACK=30 or 60.
            # If our model expects 450 inputs and we have 15 features => 450/15 = 30.
            # So LOOKBACK is 30.
            # If we had 60 lookback, inputs would be 900.
            # Let's derive lookback if possible or assume 30.
            
            check_lookback = LOOKBACK
            
            if model_type == 'ml' and hasattr(model, 'n_features_in_'):
                n_expected = model.n_features_in_
                n_feats = len(selected_cols)
                if n_feats > 0:
                    calculated_lookback = n_expected // n_feats
                    if calculated_lookback != LOOKBACK:
                        print(f"    Note: Model implies LOOKBACK={calculated_lookback} (Expected {n_expected} / {n_feats} features). Adjusting.")
                        check_lookback = calculated_lookback
                        
            # Create Sliding Windows
            X_windows = []
            
            # We need valid windows.
            # Data: [0, 1, 2, ... N]
            # Window 0: [0:30] -> Predict for 30 (or 29?)
            # Usually we predict the NEXT step after the window.
            # Training: X = data[i-LB:i], y = label[i]
            
            if len(feature_data) <= check_lookback:
                print("    Not enough data for lookback.")
                continue
                
            for i in range(check_lookback, len(feature_data)):
                window = feature_data[i-check_lookback:i]
                
                if model_type == 'ml':
                    # Flatten for ML
                    X_windows.append(window.flatten())
                else:
                    # Keep 3D for Keras (LSTM/CNN)
                    X_windows.append(window)
                    
            X_test = np.array(X_windows)
            
            # Align Prices/Dates to Predictions
            # Prediction at index i corresponds to data window ending at i.
            # The "price" at this moment is price[i-1] (end of window)?
            # Or are we predicting FOR the next candle?
            # Standard: At time T, we use window [T-L:T] to predict T+1 or direction at T.
            # Let's assume we predict for the candle at 'i'. 
            # So alignment:
            # X_test[0] uses data[0:30]. This corresponds to index 30.
            # The signal generated is for the period STARTING at 30?
            
            # IMPORTANT: The simulation loop uses `signals` and `prices`.
            # We need `signals` to verify against `prices`.
            # signals[k] corresponds to price movement from k to k+1?
            
            # Let's align arrays.
            # valid_indices starts at `check_lookback`.
            valid_indices = range(check_lookback, len(feature_data))
            dates = df.index[valid_indices]
            prices = df['Close'].iloc[valid_indices].values
            
            # Scaling
            if scaler:
                try:
                    if model_type == 'ml':
                        # ML models might have been trained on SCALED inputs?
                        # Or scaled windows?
                        # Usually scaler is per-feature.
                        # If we have 1D array of 450 feats, distinct from 15 features scaler.
                        # BUT `scaler.pkl` usually stores 15-feature scaler.
                        # So we should scale the WINDOWS?
                        # Correct approach: Scale the raw data FIRST, then window/flatten.
                        
                        # Retrying: Scale raw data first
                        feature_data_scaled = scaler.transform(feature_data)
                        
                        # Re-create windows from scaled data
                        X_windows_scaled = []
                        for i in range(check_lookback, len(feature_data_scaled)):
                            window = feature_data_scaled[i-check_lookback:i]
                            X_windows_scaled.append(window.flatten())
                        X_test_scaled = np.array(X_windows_scaled)
                        
                    else:
                        # Keras matches 3D
                         feature_data_scaled = scaler.transform(feature_data)
                         X_windows_scaled = []
                         for i in range(check_lookback, len(feature_data_scaled)):
                            window = feature_data_scaled[i-check_lookback:i]
                            X_windows_scaled.append(window)
                         X_test_scaled = np.array(X_windows_scaled)
                         
                except ValueError as e:
                    print(f"    Error scaling: {e}.")
                    continue
            else:
                X_test_scaled = X_test

            try:
                probs = None
                if model_type == 'keras':
                    probs = model.predict(X_test_scaled, verbose=0)
                else:
                    # ML models
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_test_scaled)
                    else:
                        signals_default = model.predict(X_test_scaled)
                        probs = np.zeros((len(signals_default), 3 if np.max(signals_default)>1 else 2))
                        for i, s in enumerate(signals_default):
                             probs[i, int(s)] = 1.0
                
                if probs is not None:
                    # Optimize Threshold
                    best_thresh = 0.50
                    best_score = -99999
                    best_stats = None
                    
                    feature_class = np.argmax(probs, axis=1)
                    max_prob = np.max(probs, axis=1)
                    
                    # Grid Search: 0.50 to 0.90 with step 0.01
                    thresholds = np.arange(0.50, 0.90, 0.01)
                    
                    print(f"  Optimizing Threshold (0.50-0.90)...")
                    
                    for th in thresholds:
                        # Construct signals
                        signals_th = []
                        # Vectorized signal construction for speed
                        # 99 = Neutral
                        neutral_mask = max_prob < th
                        signals_th = np.where(neutral_mask, 99, feature_class)
                        
                        eq_curve = run_simulation(signals_th, prices)
                        if not eq_curve: continue
                        
                        final_val = eq_curve[-1]
                        ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                        
                        # Score: Simple Return
                        score = ret 
                        
                        if score > best_score:
                            best_score = score
                            best_thresh = th
                            best_stats = (ret, calculate_drawdown(pd.Series(eq_curve)), final_val, signals_th, eq_curve)
                            trades_count = np.sum(signals_th != 99)
                            # Only print if we found a better settings
                            if abs(ret) > 0.01: # Filter out 0.00% changes if cleaner output desired, or just print all improvements
                                 print(f"    New Best: Thresh {th:.2f} -> Ret {ret:+.2f}% | Trades {trades_count}")
                            
                    # Finalize with Best Threshold
                    if best_stats:
                        total_return, max_dd, final_equity, signals, equity_curve = best_stats
                        print(f"  > SELECTED Best Threshold: {best_thresh:.2f} (Ret: {total_return:.2f}%)")
                    else:
                        print("  > Optimization failed. Using default.")
                        # Fallback logic if needed
                        continue
                        
                    # Debug: Stats
                    n_trades = np.sum(np.array(signals) != 99)
                    avg_conf = np.mean(max_prob)
                    print(f"  > Signals: {n_trades} trades | {len(signals)-n_trades} skipped (Avg Conf: {avg_conf:.2f})")
                    
                # 5. Simulate (Already done in optimization, just plotting now)
                # equity_curve is already from best_stats
                
                bnh_return = (prices[-1] - prices[0]) / prices[0] * 100
                
                print(f"  Result: {total_return:+.2f}% (B&H: {bnh_return:+.2f}%) | DD: {max_dd:.2f}%")
                
                # Plot
                plot_backtest(dates, equity_curve, prices, signals, market, trend, model_name, total_return, bnh_return)
                
                results.append({
                    'Market': market,
                    'Trend': trend,
                    'Regime': regime_method,
                    'Champion Model': model_name,
                    'Best Threshold': best_thresh,
                    'Return (%)': round(total_return, 2),
                    'B&H (%)': round(bnh_return, 2),
                    'Max DD (%)': round(max_dd, 2),
                    'Final $': round(final_equity, 2)
                })
                
            except Exception as e:
                 print(f"  Simulation Error: {e}")
            
            # Clear session to prevent TF retracing warnings
            if model_type == 'keras':
                tf.keras.backend.clear_session()

    # Save to CSV
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv("backtest_champion_results.csv", index=False)
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(res_df.to_string(index=False))
        print("\nSaved to backtest_champion_results.csv")
    else:
        print("\nNo results generated.")


def main_combined():
    """
    Combined Backtest: Uses Regime Detection to switch between Uptrend and Downtrend models.
    Uses MOO-Optimized Parameters (per market) for confidence, thresholds, and stop-loss.
    Outputs ONE result per market (instead of separate uptrend/downtrend rows).
    """
    print("=" * 70)
    print("COMBINED BACKTEST: REGIME-SWITCHING MODELS (MOO-Optimized)")
    print("=" * 70)
    
    # Load best prediction models using MOO composite score
    print("\n[Loading Prediction Models - MOO Multi-Objective Selection]")
    best_models_map = load_best_models_map_moo()
    if not best_models_map:
        print("No champion models found.")
        return

    # Load best regime methods
    print("\n[Loading Regime Detection Methods from CSV]")
    best_regime_map = load_best_regime_methods()
    
    # Print MOO-Optimized Parameters
    print("\n[MOO-Optimized Parameters]")
    for mkt, params in OPTIMIZED_PARAMS.items():
        print(f"  {mkt}: confidence={params['confidence']:.4f}, long={params['long']:.4f}, short={params['short']:.4f}, stop_loss={params['stop_loss']:.4f}")
    
    results_original = []
    results_moo = []
    results_moo = []
    
    for market in MARKETS:
        print(f"\n{'='*60}")
        print(f">> MARKET: {market}")
        print("="*60)
        
        # Get MOO params for this market
        moo_params = OPTIMIZED_PARAMS.get(market, {
            'confidence': 0.55, 'long': 0.54, 'short': 0.46, 'stop_loss': 0.05
        })
        print(f"   MOO Params: conf={moo_params['confidence']:.4f}, long={moo_params['long']:.4f}, short={moo_params['short']:.4f}, SL={moo_params['stop_loss']:.4f}")
        
        # Get regime method for this market
        regime_method = best_regime_map.get(market, 'HMM')
        print(f"   Regime Method: {regime_method}")
        
        # Get uptrend and downtrend model names
        uptrend_model_name = best_models_map.get((market, 'uptrend'))
        downtrend_model_name = best_models_map.get((market, 'downtrend'))
        
        if not uptrend_model_name or not downtrend_model_name:
            print(f"   Missing models for {market}. Skipping.")
            continue
            
        print(f"   Uptrend Model: {uptrend_model_name}")
        print(f"   Downtrend Model: {downtrend_model_name}")
        
        # 1. Load Data
        ticker = get_ticker(market)
        if not ticker:
            continue
            
        print(f"   Loading data for {ticker}...")
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if df.empty:
                print("   Empty data. Skipping.")
                continue
                
            df = add_technical_features(df)
            df = df.dropna()
            
        except Exception as e:
            print(f"   Error loading data: {e}")
            continue
        
        # 2. Run Regime Detection
        print(f"   Running Regime Detection ({regime_method})...")
        try:
            if regime_method == 'HMM':
                regime = RegimeDetector.detect_hmm(df, market_name=market)
            elif regime_method == 'HMM_Enhanced':
                regime = RegimeDetector.detect_hmm_enhanced(df)
            elif regime_method == 'GMM':
                regime = RegimeDetector.detect_gmm(df, market_name=market)
            elif regime_method == 'GMM_Enhanced':
                regime = RegimeDetector.detect_gmm_enhanced(df)
            elif regime_method == 'SMA200':
                regime = RegimeDetector.detect_sma200(df)
            else:
                regime = RegimeDetector.detect_hmm(df, market_name=market)
        except Exception as e:
            print(f"   Regime Detection Error: {e}. Using SMA200 fallback.")
            regime = (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values
        
        uptrend_days = np.sum(regime == 1)
        downtrend_days = np.sum(regime == 0)
        print(f"   Regime: {uptrend_days} uptrend days, {downtrend_days} downtrend days")
        
        # 3. Load Both Models
        uptrend_dir = os.path.join(MODEL_BASE_DIR, f"model_{market}_uptrend")
        downtrend_dir = os.path.join(MODEL_BASE_DIR, f"model_{market}_downtrend")
        
        try:
            up_model, up_scaler, up_le, up_type = load_model_and_objects(uptrend_dir, uptrend_model_name)
            down_model, down_scaler, down_le, down_type = load_model_and_objects(downtrend_dir, downtrend_model_name)
        except Exception as e:
            print(f"   Error loading models: {e}")
            continue
            
        if not up_model or not down_model:
            print("   Model loading failed. Skipping.")
            continue
        
        # 4. Prepare Features
        selected_cols = get_selected_features(df)
        if not selected_cols:
            print("   No features found. Skipping.")
            continue
            
        missing = [f for f in selected_cols if f not in df.columns]
        for m in missing:
            df[m] = 0
            
        feature_data = df[selected_cols].values
        
        # Determine lookback
        check_lookback = LOOKBACK
        if up_type == 'ml' and hasattr(up_model, 'n_features_in_'):
            n_expected = up_model.n_features_in_
            n_feats = len(selected_cols)
            if n_feats > 0:
                check_lookback = n_expected // n_feats
        
        if len(feature_data) <= check_lookback:
            print("   Not enough data for lookback. Skipping.")
            continue
        
        # 5. Generate Combined Signals
        print("   Generating combined signals...")
        
        # Scale data
        if up_scaler:
            try:
                feature_data_scaled = up_scaler.transform(feature_data)
            except:
                feature_data_scaled = feature_data
        else:
            feature_data_scaled = feature_data
        
        # Create windows
        X_windows_up = []
        X_windows_down = []
        
        for i in range(check_lookback, len(feature_data_scaled)):
            window = feature_data_scaled[i-check_lookback:i]
            if up_type == 'ml':
                X_windows_up.append(window.flatten())
            else:
                X_windows_up.append(window)
                
            if down_type == 'ml':
                X_windows_down.append(window.flatten())
            else:
                X_windows_down.append(window)
        
        X_up = np.array(X_windows_up)
        X_down = np.array(X_windows_down)
        
        # Get predictions from both models
        try:
            if up_type == 'keras':
                probs_up = up_model.predict(X_up, verbose=0)
            else:
                if hasattr(up_model, 'predict_proba'):
                    probs_up = up_model.predict_proba(X_up)
                else:
                    preds = up_model.predict(X_up)
                    probs_up = np.zeros((len(preds), 2))
                    for i, p in enumerate(preds):
                        probs_up[i, int(p)] = 1.0
                        
            if down_type == 'keras':
                probs_down = down_model.predict(X_down, verbose=0)
            else:
                if hasattr(down_model, 'predict_proba'):
                    probs_down = down_model.predict_proba(X_down)
                else:
                    preds = down_model.predict(X_down)
                    probs_down = np.zeros((len(preds), 2))
                    for i, p in enumerate(preds):
                        probs_down[i, int(p)] = 1.0
                        
        except Exception as e:
            print(f"   Prediction error: {e}")
            continue
        
        # Combine signals based on regime
        valid_indices = range(check_lookback, len(feature_data))
        regime_aligned = regime[list(valid_indices)]
        prices = df['Close'].iloc[list(valid_indices)].values
        dates = df.index[list(valid_indices)]
        
        # ──────────────────────────────────────────────
        # A) ORIGINAL signals (fixed thresholds)
        # ──────────────────────────────────────────────
        original_signals = []
        original_conf_thresh = 0.55  # Original default
        original_long = 0.54
        original_short = 0.46
        
        for i in range(len(regime_aligned)):
            if regime_aligned[i] == 1:
                bull_prob = probs_up[i, 1] if probs_up.shape[1] > 1 else probs_up[i, 0]
                conf = np.max(probs_up[i])
            else:
                bull_prob = probs_down[i, 1] if probs_down.shape[1] > 1 else probs_down[i, 0]
                conf = np.max(probs_down[i])
            
            if conf < original_conf_thresh:
                original_signals.append(99)
            elif bull_prob >= original_long:
                original_signals.append(1)
            elif bull_prob <= original_short:
                original_signals.append(0)
            else:
                original_signals.append(99)
        
        signals_orig = np.array(original_signals)
        
        # ──────────────────────────────────────────────
        # B) WALK-FORWARD GRID SEARCH — No Look-Ahead Bias
        # ──────────────────────────────────────────────
        bnh_return = (prices[-1] - prices[0]) / prices[0] * 100
        
        # A) Original (fixed thresholds)
        eq_orig = run_simulation(signals_orig, prices)
        if eq_orig:
            ret_orig = (eq_orig[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            dd_orig = calculate_drawdown(pd.Series(eq_orig))
            trades_orig = int(np.sum(signals_orig != 99))
        else:
            ret_orig, dd_orig, trades_orig = 0.0, 0.0, 0
            eq_orig = [INITIAL_CAPITAL]
        
        # B) Walk-Forward Optimization
        print(f"   ⚡ Running Walk-Forward Grid Search for {market}...")
        
        # Parameter grid
        conf_grid = [0.0, 0.50, 0.55, 0.60]
        long_grid = [0.50, 0.55, 0.60]
        short_grid = [0.40, 0.45, 0.50]
        sl_grid = [0.03, 0.08, 0.15, 0.25, 0.50]
        strategy_grid = ['active', 'smart_hold']
        trailing_grid = [0.0, 0.10, 0.20]
        long_only_options = [True, False]
        fallback_options = [True, False]
        leverage_grid = [1.0, 1.5, 2.0]
        sizing_mode_grid = ['conservative', 'moderate', 'aggressive']
        min_position_grid = [0.0, 0.3, 0.5]
        
        # Walk-Forward windows
        n_total = len(prices)
        train_days = min(252, n_total // 2)  # ~1 year or half data
        test_days = 63   # ~3 months
        
        oos_equity_parts = []
        oos_trade_points_all = []
        window_params_list = []
        oos_start_capital = INITIAL_CAPITAL
        total_oos_trades = 0
        total_oos_wins = 0
        n_windows = 0
        total_combos_all = 0
        
        def _gen_signals(regime_arr, p_up, p_down, conf_t, long_t, short_t, fb):
            sigs, confs = [], []
            for i in range(len(regime_arr)):
                if regime_arr[i] == 1:
                    bp = p_up[i, 1] if p_up.shape[1] > 1 else p_up[i, 0]
                    c = np.max(p_up[i])
                else:
                    bp = p_down[i, 1] if p_down.shape[1] > 1 else p_down[i, 0]
                    c = np.max(p_down[i])
                confs.append(c)
                if c < conf_t:
                    sigs.append(regime_arr[i] if fb else 99)
                elif bp >= long_t:
                    sigs.append(1)
                elif bp <= short_t:
                    sigs.append(0)
                else:
                    sigs.append(regime_arr[i] if fb else 99)
            return np.array(sigs), np.array(confs)
        
        w_start = 0
        while w_start + train_days + test_days <= n_total:
            train_end = w_start + train_days
            test_end = min(train_end + test_days, n_total)
            
            # Slice data
            tr_prices = prices[w_start:train_end]
            tr_regime = regime_aligned[w_start:train_end]
            tr_pup = probs_up[w_start:train_end]
            tr_pdn = probs_down[w_start:train_end]
            
            te_prices = prices[train_end:test_end]
            te_regime = regime_aligned[train_end:test_end]
            te_pup = probs_up[train_end:test_end]
            te_pdn = probs_down[train_end:test_end]
            
            # Grid Search on TRAIN only
            best_score_w = -10000
            best_params_w = {}
            total_combos = 0
            
            for conf_t in conf_grid:
                for long_t in long_grid:
                    for short_t in short_grid:
                        if short_t >= long_t:
                            continue
                        for fallback_opt in fallback_options:
                            for leverage_opt in leverage_grid:
                                tr_sigs, tr_confs = _gen_signals(tr_regime, tr_pup, tr_pdn, conf_t, long_t, short_t, fallback_opt)
                                
                                for sl in sl_grid:
                                    for strat in strategy_grid:
                                        for ts in trailing_grid:
                                            if strat == 'smart_hold' and ts == 0:
                                                continue
                                            for lo in long_only_options:
                                              for sm in sizing_mode_grid:
                                                for mp in min_position_grid:
                                                    total_combos += 1
                                                    eq, trades, wr, _ = run_simulation_moo(
                                                        tr_sigs, tr_prices,
                                                        confidences=tr_confs, regimes=tr_regime,
                                                        stop_loss_pct=sl, trailing_stop_pct=ts,
                                                        long_only=lo, strategy_mode=strat,
                                                        leverage=leverage_opt, sizing_mode=sm,
                                                        min_position=mp
                                                    )
                                                    if eq:
                                                        ret = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                                                        dd = calculate_drawdown(pd.Series(eq))
                                                        score = ret - (10 if dd < -50 else 0)
                                                        if score > best_score_w:
                                                            best_score_w = score
                                                            best_params_w = {
                                                                'confidence': conf_t, 'long': long_t, 'short': short_t,
                                                                'stop_loss': sl, 'strategy': strat, 'trailing': ts,
                                                                'long_only': lo, 'fallback': fallback_opt,
                                                                'leverage': leverage_opt, 'sizing_mode': sm,
                                                                'min_position': mp
                                                            }
            
            total_combos_all += total_combos
            
            # Apply best params on TEST (out-of-sample)
            if best_params_w:
                window_params_list.append(best_params_w)
                te_sigs, te_confs = _gen_signals(te_regime, te_pup, te_pdn,
                    best_params_w['confidence'], best_params_w['long'], best_params_w['short'], best_params_w['fallback'])
                
                oos_eq, oos_trades, oos_wr, oos_tp = run_simulation_moo(
                    te_sigs, te_prices,
                    confidences=te_confs, regimes=te_regime,
                    stop_loss_pct=best_params_w['stop_loss'],
                    trailing_stop_pct=best_params_w['trailing'],
                    long_only=best_params_w['long_only'],
                    strategy_mode=best_params_w['strategy'],
                    initial_capital=oos_start_capital,
                    leverage=best_params_w['leverage'],
                    sizing_mode=best_params_w['sizing_mode'],
                    min_position=best_params_w['min_position']
                )
                
                if oos_eq:
                    for idx, action in oos_tp:
                        oos_trade_points_all.append((train_end + idx, action))
                    oos_equity_parts.append(oos_eq)
                    oos_start_capital = oos_eq[-1]
                    total_oos_trades += oos_trades
                    if oos_trades > 0:
                        total_oos_wins += int(oos_wr / 100 * oos_trades)
                    
                    oos_ret = (oos_eq[-1] / (oos_eq[0] if oos_eq[0] > 0 else INITIAL_CAPITAL) - 1) * 100
                    print(f"      W{n_windows+1}: Train[{w_start}-{train_end}] Test[{train_end}-{test_end}] "
                          f"TrainBest={best_score_w:.1f}% OOS={oos_ret:.1f}% "
                          f"strat={best_params_w['strategy']}, lev={best_params_w['leverage']}, sm={best_params_w['sizing_mode']}, mp={best_params_w['min_position']}")
            
            n_windows += 1
            w_start += test_days
        
        # Handle remaining data with last params
        if w_start < n_total and window_params_list:
            last_p = window_params_list[-1]
            rem_prices = prices[w_start:]
            rem_regime = regime_aligned[w_start:]
            rem_pup = probs_up[w_start:]
            rem_pdn = probs_down[w_start:]
            
            rem_sigs, rem_confs = _gen_signals(rem_regime, rem_pup, rem_pdn,
                last_p['confidence'], last_p['long'], last_p['short'], last_p['fallback'])
            
            rem_eq, rem_trades, rem_wr, rem_tp = run_simulation_moo(
                rem_sigs, rem_prices,
                confidences=rem_confs, regimes=rem_regime,
                stop_loss_pct=last_p['stop_loss'], trailing_stop_pct=last_p['trailing'],
                long_only=last_p['long_only'], strategy_mode=last_p['strategy'],
                initial_capital=oos_start_capital,
                leverage=last_p['leverage'], sizing_mode=last_p['sizing_mode'],
                min_position=last_p['min_position']
            )
            
            if rem_eq:
                for idx, action in rem_tp:
                    oos_trade_points_all.append((w_start + idx, action))
                oos_equity_parts.append(rem_eq)
                total_oos_trades += rem_trades
                if rem_trades > 0:
                    total_oos_wins += int(rem_wr / 100 * rem_trades)
        
        # Select final params = most common across windows (mode)
        from collections import Counter
        if window_params_list:
            param_keys = ['strategy', 'leverage', 'sizing_mode', 'min_position', 'stop_loss',
                         'trailing', 'long_only', 'fallback', 'confidence', 'long', 'short']
            param_tuples = [tuple(p.get(k, None) for k in param_keys) for p in window_params_list]
            mode_tuple = Counter(param_tuples).most_common(1)[0][0]
            best_params = dict(zip(param_keys, mode_tuple))
        else:
            best_params = {'confidence': 0.55, 'long': 0.54, 'short': 0.46, 'stop_loss': 0.05,
                          'strategy': 'active', 'trailing': 0.0, 'long_only': True,
                          'fallback': True, 'leverage': 1.0, 'sizing_mode': 'conservative', 'min_position': 0.0}
        
        # Log OOS return for reference
        if oos_equity_parts:
            oos_final = oos_equity_parts[-1][-1] if oos_equity_parts[-1] else INITIAL_CAPITAL
            oos_ret = (oos_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            print(f"   📝 Walk-Forward OOS Return: {oos_ret:.2f}% ({n_windows} windows)")
        
        print(f"   📊 Final Params (mode): {best_params}")
        
        # Generate full signals using best params (same as trading_system.py)
        best_signals, best_confidences = _gen_signals(regime_aligned, probs_up, probs_down,
            best_params['confidence'], best_params['long'], best_params['short'], best_params['fallback'])
        
        # Run full-period simulation with best params (matches trading_system.py)
        eq_moo, trades_moo, winrate_moo, trade_points_moo = run_simulation_moo(
            best_signals, prices,
            confidences=best_confidences, regimes=regime_aligned,
            stop_loss_pct=best_params['stop_loss'],
            trailing_stop_pct=best_params['trailing'],
            long_only=best_params['long_only'],
            strategy_mode=best_params['strategy'],
            initial_capital=INITIAL_CAPITAL,
            leverage=best_params['leverage'],
            sizing_mode=best_params['sizing_mode'],
            min_position=best_params['min_position']
        )
        
        if not eq_moo:
            eq_moo = [INITIAL_CAPITAL]
        
        ret_moo = (eq_moo[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        dd_moo = calculate_drawdown(pd.Series(eq_moo))
        
        print(f"   ✅ Full-Period Return: {ret_moo:.2f}% (Walk-Forward selected params)")
        
        signals_moo = best_signals
        strategy_mode = best_params.get('strategy', 'unknown')
        
        # Print comparison
        print(f"\n   {'Metric':<20} {'B&H':>10} {'Original':>12} {'MOO(Hyb)':>12}")
        print(f"   {'-'*56}")
        print(f"   {'Return (%)':<20} {bnh_return:>10.2f} {ret_orig:>12.2f} {ret_moo:>12.2f}")
        print(f"   {'Max DD (%)':<20} {'':>10} {dd_orig:>12.2f} {dd_moo:>12.2f}")
        print(f"   {'Trades':<20} {'':>10} {trades_orig:>12d} {trades_moo:>12d}")
        print(f"   {'Win Rate (%)':<20} {'':>10} {'N/A':>12} {winrate_moo:>12.1f}")
        print(f"   {'vs B&H':<20} {'':>10} {ret_orig-bnh_return:>+12.2f} {ret_moo-bnh_return:>+12.2f}")
        
        results_original.append({
            'Market': market,
            'Regime Method': regime_method,
            'Uptrend Model': uptrend_model_name,
            'Downtrend Model': downtrend_model_name,
            'Type': 'Original',
            'Confidence': original_conf_thresh,
            'Threshold Long': original_long,
            'Threshold Short': original_short,
            'Stop Loss': 0.0,
            'Return (%)': round(ret_orig, 2),
            'B&H (%)': round(bnh_return, 2),
            'Max DD (%)': round(dd_orig, 2),
            'Trades': trades_orig,
            'Win Rate (%)': 0.0,
            'Final $': round(eq_orig[-1], 2)
        })
        
        results_moo.append({
            'Market': market,
            'Regime Method': regime_method,
            'Uptrend Model': uptrend_model_name,
            'Downtrend Model': downtrend_model_name,
            'Type': f'MOO ({best_params.get("strategy", "grid")})[FB:{best_params.get("fallback", False)}][LO:{best_params.get("long_only", True)}][Lev:{best_params.get("leverage", 1.0)}][SM:{best_params.get("sizing_mode", "conservative")}][MP:{best_params.get("min_position", 0.0)}]',
            'Confidence': best_params.get('confidence', 0),
            'Threshold Long': best_params.get('long', 0),
            'Threshold Short': best_params.get('short', 0),
            'Stop Loss': best_params.get('stop_loss', 0),
            'Trailing': best_params.get('trailing', 0),
            'Return (%)': round(ret_moo, 2),
            'B&H (%)': round(bnh_return, 2),
            'Max DD (%)': round(dd_moo, 2),
            'Trades': trades_moo,
            'Win Rate (%)': round(winrate_moo, 1),
            'Final $': round(eq_moo[-1], 2)
        })

        # Plot the Hybrid Strategy
        plot_backtest(dates, eq_moo, prices, signals_moo, market, "Combined", "Hybrid_MOO", ret_moo, bnh_return, regime=regime_aligned)

        # Clear Keras session
        tf.keras.backend.clear_session()
    
    # Save Results
    all_results = results_original + results_moo
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv("backtest_combined_results.csv", index=False)
        
        print("\n" + "="*70)
        print("FINAL COMBINED RESULTS (Hybrid Strategy: Active vs Smart Hold)")
        print("="*70)
        print(res_df.to_string(index=False))
        
        # Summary comparison
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Market':<8} {'B&H':>10} {'Orig':>10} {'MOO(Hyb)':>10} {'Orig vs B&H':>12} {'MOO vs B&H':>12}")
        print("-" * 80)
        
        summary_rows = []
        for o, m in zip(results_original, results_moo):
            orig_vs_bh = o['Return (%)'] - o['B&H (%)']
            moo_vs_bh = m['Return (%)'] - m['B&H (%)']
            print(f"{o['Market']:<8} {o['B&H (%)']:>10.2f} {o['Return (%)']:>10.2f} {m['Return (%)']:>10.2f} {orig_vs_bh:>+12.2f} {moo_vs_bh:>+12.2f}")
            
            summary_rows.append({
                'Market': o['Market'],
                'B&H': round(o['B&H (%)'], 2),
                'Orig': round(o['Return (%)'], 2),
                'MOO(Hyb)': round(m['Return (%)'], 2),
                'Orig vs B&H': round(orig_vs_bh, 2),
                'MOO vs B&H': round(moo_vs_bh, 2)
            })
            
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = "backtest_comparison_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        print(f"\nSaved to backtest_combined_results.csv")
        print(f"Saved summary to {summary_csv_path}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--combined':
        main_combined()
    else:
        main()

