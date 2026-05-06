"""
All-Market Adaptive Trading System
- Fetches data from yfinance (max history to present)
- Detects current trend for each market
- Selects best model based on market + trend
- Generates trading signals
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../model'))

from config import TICKERS, BEST_MODELS, LOOKBACK, BINARY_MODE, TREND_METHOD
from model.features import calculate_features, get_selected_features
from model.regime_detection import RegimeDetector

class AdaptiveTrader:
    """
    Adaptive trading system that selects the best model based on:
    1. Current market being traded
    2. Current trend (uptrend/downtrend) of that market
    """
    
    def __init__(self):
        self.models = {}  # Cache loaded models
        self.scalers = {}
        
    def fetch_data(self, market: str) -> pd.DataFrame:
        """Fetch historical data from yfinance"""
        ticker = TICKERS.get(market)
        if not ticker:
            print(f"Unknown market: {market}")
            return pd.DataFrame()
            
        print(f"Fetching {market} ({ticker}) data...")
        
        try:
            df = yf.download(ticker, period="max", progress=False)
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            print(f"  Downloaded {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
            return df
            
        except Exception as e:
            print(f"Error fetching {market}: {e}")
            return pd.DataFrame()
    
    def detect_trend(self, df: pd.DataFrame, market: str) -> str:
        """
        Detect current trend using Market-Specific Methods based on backtest analysis.
        
        Best Methods:
        - US, UK, Gold (Mature): Standard HMM (Pure Returns) -> Best Separation/Excess Return.
        - Thai (Emerging): Enhanced HMM (Returns + Volatility) -> Best Separation.
        - BTC (Crypto): Enhanced GMM (Returns + Volatility + RSI + ADX) -> Best Separation.
        """
        if len(df) < 200:
            return 'uptrend'
            
        # Market-Specific Selection
        if market in ['US', 'UK', 'Gold']:
            # Mature markets: Standard HMM works best
            is_uptrend = RegimeDetector.detect_hmm(df)
            
        elif market == 'Thai':
            # Emerging market: Enhanced HMM (Vol sensitivity) works best
            is_uptrend = RegimeDetector.detect_hmm_enhanced(df)
            
        elif market == 'BTC':
            # Crypto: Enhanced GMM (Momentum/Vol features) works best
            is_uptrend = RegimeDetector.detect_gmm_enhanced(df)
            
        else:
            # Default fallback for unknown markets
            is_uptrend = RegimeDetector.detect_hmm(df)

        # Get current trend (last value)
        current_trend = 'uptrend' if is_uptrend[-1] == 1 else 'downtrend'
        return current_trend
    
    def load_model(self, market: str, trend: str):
        """Load the best model for given market and trend"""
        model_name = BEST_MODELS.get(market, {}).get(trend)
        if not model_name:
            print(f"No model configured for {market}/{trend}")
            return None
            
        cache_key = f"{market}_{trend}_{model_name}"
        if cache_key in self.models:
            return self.models[cache_key]
            
        # Determine model path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, f"model_{trend}")
        
        # Try .keras first, then .pkl
        keras_path = os.path.join(model_dir, f"{model_name}.keras")
        pkl_path = os.path.join(model_dir, f"{model_name}.pkl")
        
        model = None
        if os.path.exists(keras_path):
            print(f"  Loading {model_name} from {keras_path}")
            model = tf.keras.models.load_model(keras_path)
        elif os.path.exists(pkl_path):
            print(f"  Loading {model_name} from {pkl_path}")
            model = joblib.load(pkl_path)
        else:
            print(f"  Model not found: {keras_path} or {pkl_path}")
            return None
            
        self.models[cache_key] = model
        return model
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate features and prepare for prediction"""
        df_feat = calculate_features(df)
        
        if df_feat.empty:
            return None
            
        feature_cols = get_selected_features(df_feat)
        data = df_feat[feature_cols].values
        
        if len(data) < LOOKBACK:
            print(f"  Not enough data for LOOKBACK={LOOKBACK}")
            return None
            
        # Get last LOOKBACK rows as sequence
        sequence = data[-LOOKBACK:]
        
        # Normalize (simple z-score)
        mean = np.nanmean(sequence, axis=0)
        std = np.nanstd(sequence, axis=0) + 1e-9
        sequence = (sequence - mean) / std
        
        # Handle NaN/Inf
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        return sequence.reshape(1, LOOKBACK, -1)
    
    def predict(self, model, X: np.ndarray, model_name: str) -> tuple:
        """Make prediction and return (signal, probability)"""
        if model is None or X is None:
            return 'UNKNOWN', 0.0
            
        try:
            if hasattr(model, 'predict_proba'):
                # Scikit-learn models need 2D input
                X_flat = X.reshape(X.shape[0], -1)
                probs = model.predict_proba(X_flat)[0]
            else:
                # Keras models
                probs = model.predict(X, verbose=0)[0]
            
            pred_class = np.argmax(probs)
            prob = probs[pred_class]
            
            if BINARY_MODE:
                signal = 'BULLISH' if pred_class == 1 else 'BEARISH'
            else:
                signals = ['BEARISH', 'NEUTRAL', 'BULLISH']
                signal = signals[pred_class]
                
            return signal, prob
            
        except Exception as e:
            print(f"  Prediction error: {e}")
            return 'ERROR', 0.0
    
    def run(self):
        """Run the trading system for all markets"""
        print("=" * 60)
        print("ALL-MARKET ADAPTIVE TRADING SYSTEM")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)
        
        results = []
        
        for market in TICKERS.keys():
            print(f"\n--- {market} ---")
            
            # 1. Fetch data
            df = self.fetch_data(market)
            if df.empty:
                continue
                
            # 2. Detect trend
            trend = self.detect_trend(df, market)
            print(f"  Current Trend: {trend.upper()}")
            
            # 3. Load best model for this market+trend
            model_name = BEST_MODELS.get(market, {}).get(trend, 'Unknown')
            print(f"  Selected Model: {model_name}")
            model = self.load_model(market, trend)
            
            if model is None:
                results.append({
                    'Market': market,
                    'Date': df.index[-1].date(),
                    'Trend': trend,
                    'Model': model_name,
                    'Signal': 'NO_MODEL',
                    'Probability': 0.0
                })
                continue
            
            # 4. Prepare features
            X = self.prepare_features(df)
            
            # 5. Predict
            signal, prob = self.predict(model, X, model_name)
            print(f"  Signal: {signal} (Probability: {prob:.2%})")
            
            results.append({
                'Market': market,
                'Date': df.index[-1].date(),
                'Trend': trend,
                'Model': model_name,
                'Signal': signal,
                'Probability': round(prob * 100, 2)
            })
        
        # Save results
        if results:
            results_df = pd.DataFrame(results)
            output_path = os.path.join(os.path.dirname(__file__), 'trading_signals.csv')
            results_df.to_csv(output_path, index=False)
            
            print("\n" + "=" * 60)
            print("TRADING SIGNALS SUMMARY")
            print("=" * 60)
            print(results_df.to_string(index=False))
            print(f"\nSaved to {output_path}")
            
        return results


def main():
    trader = AdaptiveTrader()
    trader.run()


if __name__ == "__main__":
    main()
