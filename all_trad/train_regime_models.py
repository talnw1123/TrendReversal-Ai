"""
Train Regime Detection Models
Trains and saves Random Forest, GMM, and HMM models for each market as .pkl files.
These models can be loaded by RegimeDetector for faster inference.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../model'))

from model.features import calculate_features

# Market tickers
TICKERS = {
    'Thai': '^SET.BK',
    'UK': '^FTSE',
    'Gold': 'GC=F',
    'US': '^GSPC',
    'BTC': 'BTC-USD'
}

# Output directory for models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'regime_models')


def train_random_forest(df_feat, market_name, valid_features):
    """Train and save Random Forest model"""
    print("  [RF] Training Random Forest...")
    
    # Create Target
    df_feat['future_ret'] = df_feat['Close'].shift(-20) / df_feat['Close'] - 1
    df_feat['target'] = (df_feat['future_ret'] > 0).astype(int)
    
    valid_mask = ~df_feat['future_ret'].isna()
    X = df_feat.loc[valid_mask, valid_features].values
    y = df_feat.loc[valid_mask, 'target'].values
    X = np.nan_to_num(X)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Save
    model_path = os.path.join(MODEL_DIR, f'regime_rf_{market_name}.pkl')
    features_path = os.path.join(MODEL_DIR, f'regime_features_{market_name}.pkl')
    
    joblib.dump(rf, model_path)
    joblib.dump(valid_features, features_path)
    
    print(f"  [RF] ✅ Saved: {model_path}")
    return rf


def train_gmm(df, market_name, window=20):
    """Train and save GMM model"""
    print("  [GMM] Training Gaussian Mixture Model...")
    
    data = df.copy()
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['volatility'] = data['log_ret'].rolling(window).std()
    data = data.dropna()
    
    if len(data) < 100:
        print("  [GMM] Not enough data")
        return None
    
    X = data[['log_ret', 'volatility']].values
    
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=10)
    gmm.fit(X)
    
    # Determine which component is uptrend (higher mean return)
    means = gmm.means_[:, 0]
    uptrend_label = np.argmax(means)
    
    # Save model and metadata
    model_path = os.path.join(MODEL_DIR, f'regime_gmm_{market_name}.pkl')
    meta_path = os.path.join(MODEL_DIR, f'regime_gmm_meta_{market_name}.pkl')
    
    joblib.dump(gmm, model_path)
    joblib.dump({'uptrend_label': uptrend_label, 'window': window}, meta_path)
    
    print(f"  [GMM] ✅ Saved: {model_path}")
    return gmm


def train_hmm(df, market_name):
    """Train and save HMM model"""
    print("  [HMM] Training Hidden Markov Model...")
    
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [HMM] ⚠️ hmmlearn not installed. Skipping HMM.")
        return None
    
    data = df.copy()
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    valid_data = data.dropna().copy()
    
    if len(valid_data) < 100:
        print("  [HMM] Not enough data")
        return None
    
    X = valid_data['log_ret'].values.reshape(-1, 1)
    
    hmm = GaussianHMM(n_components=2, covariance_type='diag', n_iter=100, random_state=42)
    try:
        hmm.fit(X)
    except Exception as e:
        print(f"  [HMM] Training error: {e}")
        return None
    
    # Determine which state is uptrend
    means = hmm.means_.flatten()
    uptrend_state = np.argmax(means)
    
    # Save
    model_path = os.path.join(MODEL_DIR, f'regime_hmm_{market_name}.pkl')
    meta_path = os.path.join(MODEL_DIR, f'regime_hmm_meta_{market_name}.pkl')
    
    joblib.dump(hmm, model_path)
    joblib.dump({'uptrend_state': uptrend_state}, meta_path)
    
    print(f"  [HMM] ✅ Saved: {model_path}")
    return hmm


def train_all_models(ticker, market_name):
    """Train all regime models for a market"""
    print(f"\n{'='*60}")
    print(f"Training Models for {market_name} ({ticker})")
    print("=" * 60)
    
    # 1. Fetch Data
    try:
        df = yf.download(ticker, period="max", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty or len(df) < 500:
        print("Not enough data.")
        return

    print(f"  Data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    
    # 2. Calculate Features for RF
    print("  Calculating features...")
    df_feat = calculate_features(df)
    
    # Load feature list
    csv_path = os.path.join(os.path.dirname(__file__), f'regime_features_{market_name}.csv')
    if os.path.exists(csv_path):
        try:
            feature_df = pd.read_csv(csv_path)
            selected_features = feature_df['Feature'].tolist()
        except:
            selected_features = []
    else:
        selected_features = [
            'volume_nvi', 'volume_obv', 'vma_120', 'volume_vpt', 'ma_120', 'vma_60',
            'p_ma_dist_60_120', 'trend_visual_ichimoku_b', 'vma_20', 'volume_adi', 'ma_60'
        ]
    
    valid_features = [f for f in selected_features if f in df_feat.columns]
    
    # Create output directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 3. Train Models
    if valid_features:
        train_random_forest(df_feat, market_name, valid_features)
    else:
        print("  [RF] ⚠️ No valid features, skipping RF")
    
    train_gmm(df, market_name)
    train_hmm(df, market_name)


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING ALL REGIME DETECTION MODELS (RF, GMM, HMM)")
    print("=" * 60)
    
    for market, ticker in TICKERS.items():
        train_all_models(ticker, market)
    
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED AND SAVED")
    print(f"Output directory: {MODEL_DIR}")
    print("=" * 60)
