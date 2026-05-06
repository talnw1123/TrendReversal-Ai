"""
Analyze Regime Features
This script uses Random Forest to determine which features from model/features.py
are most important for predicting market regimes (Uptrend/Downtrend).
It analyzes 5 markets (Thai, UK, Gold, US, BTC) and saves filtered feature lists to CSV.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../model'))

from model.features import calculate_features

def analyze_features_for_market(ticker, market_name):
    print(f"\n--- Analyzing Features for {market_name} ({ticker}) ---")
    
    # 1. Fetch Data
    try:
        df = yf.download(ticker, period="max", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    if df.empty or len(df) < 500:
        print("Not enough data.")
        return None

    # 2. Calculate Features
    print("Calculating features...")
    df_feat = calculate_features(df)
    
    # 3. Create Target (Regime)
    # Target: Next 20 days return > 0 (Bull Market)
    df_feat['future_ret'] = df_feat['Close'].shift(-20) / df_feat['Close'] - 1
    df_feat['target'] = (df_feat['future_ret'] > 0).astype(int)
    
    # Drop NaNs created by shifting and feature calculation
    df_feat = df_feat.dropna()
    
    # Select numeric features only (exclude target/future_ret)
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ['target', 'future_ret', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    
    # 4. Train Random Forest
    X = df_feat[feature_cols]
    y = df_feat['target']
    
    # Check for Inf values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Training Random Forest on {len(X)} samples with {len(feature_cols)} features...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X, y)
    
    # 5. Extract Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    sorted_features = []
    print("\nTop 10 Features for Regime Detection:")
    for i in range(len(feature_cols)):
        feature_name = feature_cols[indices[i]]
        importance = importances[indices[i]]
        sorted_features.append((feature_name, importance))
        if i < 10:
             print(f"{i+1}. {feature_name:30} ({importance:.4f})")
        
    return sorted_features

if __name__ == "__main__":
    # Analyze all markets
    markets = {
        'Thai': '^SET.BK',
        'UK': '^FTSE',
        'Gold': 'GC=F',
        'US': '^GSPC',
        'BTC': 'BTC-USD'
    }
    
    all_top_features = []
    
    for name, ticker in markets.items():
        # returns list of (feature, importance) tuples
        features_with_scores = analyze_features_for_market(ticker, name)
        
        if features_with_scores:
            # Create DataFrame
            df_res = pd.DataFrame(features_with_scores, columns=['Feature', 'Importance'])
            
            # Filter: Keep ONLY "really good" features
            # Strategy: 
            # 1. Importance MUST be > 1.2x Average (Significantly better than random noise)
            # 2. Cap at Top 30 features max (Conciseness)
            threshold = df_res['Importance'].mean() * 1.2
            df_filtered = df_res[df_res['Importance'] > threshold].head(30)
            
            print(f"  Kept {len(df_filtered)} features (Threshold: {threshold:.5f}, Max: 30)")
            
            # Save to CSV
            output_file = os.path.join(os.path.dirname(__file__), f'regime_features_{name}.csv')
            df_filtered.to_csv(output_file, index=False)
            print(f"  Saved to {output_file}")
            
            # Collect top 20 for consensus check
            all_top_features.extend(df_res.head(20)['Feature'].tolist())
            
    # Find consensus (most frequent top features)
    print("\n" + "="*50)
    print("CONSENSUS TOP FEATURES (Across All Markets)")
    print("="*50)
    counts = Counter(all_top_features)
    for feature, count in counts.most_common(20):
        print(f"{feature}: {count} markets")
