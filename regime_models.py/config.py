"""
Configuration for All-Market Adaptive Trading System
"""

# Ticker symbols for each market
TICKERS = {
    'Thai': '^SET.BK',   # SET Index
    'UK': '^FTSE',       # FTSE 100
    'Gold': 'GC=F',      # Gold Futures
    'US': '^GSPC',       # S&P 500
    'BTC': 'BTC-USD'     # Bitcoin
}

# Best models for each market and trend (from separate_models_comparison.csv)
# Best models for each market and trend (from separate_models_comparison.csv)
BEST_MODELS = {
    'Thai': {
        'uptrend': 'Transformer',    # 61.87%
        'downtrend': 'LSTM'          # 67.0%
    },
    'UK': {
        'uptrend': 'Transformer',    # 69.94%
        'downtrend': 'SVM'           # 57.38% (Better than RandomForest 57.05%)
    },
    'Gold': {
        'uptrend': 'Transformer',    # 68.32%
        'downtrend': 'CNN'           # 64.64% (Next best after missing Ensemble)
    },
    'US': {
        'uptrend': 'Transformer',    # 65.71%
        'downtrend': 'Transformer'   # 54.55%
    },
    'BTC': {
        'uptrend': 'SVM',            # 63.28%
        'downtrend': 'MLP'           # 61.67%
    }
}

# Use HMM for all markets (best overall performance)
TREND_METHOD = 'hmm'

# Which markets should use the model vs Buy & Hold
# True = Use adaptive model, False = Use Buy & Hold (always long)
USE_MODEL = {
    'Thai': True,
    'UK': True,
    'Gold': True,
    'US': True,
    'BTC': True
}

# Model parameters (must match training)
LOOKBACK = 30
BINARY_MODE = True

# Trading parameters
CONFIDENCE_THRESHOLD = 0.50
STOP_LOSS_PCT = 0.05


