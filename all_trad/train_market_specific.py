"""
Train Market-Specific Models for UK, Gold, BTC
- Downloads data for each market
- Creates labels based on future returns
- Splits by trend (HMM)
- Trains LSTM, CNN, Transformer
- Saves best model
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../model'))

from model.features import calculate_features, get_selected_features
from model.regime_detection import RegimeDetector

# Markets to train
MARKETS_TO_TRAIN = {
    'UK': '^FTSE',
    'Gold': 'GC=F',
    'BTC': 'BTC-USD'
}

LOOKBACK = 30
EPOCHS = 50
BATCH_SIZE = 32
LABEL_HORIZON = 5  # Days ahead for labeling


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_lstm_model(input_shape, num_classes=2):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


def create_cnn_model(input_shape, num_classes=2):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


def create_transformer_model(input_shape, num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = Dense(64)(inputs)
    x = TransformerBlock(64, 4, 128)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


def prepare_data(market: str, ticker: str):
    """Download and prepare data for a specific market"""
    print(f"\n{'='*60}")
    print(f"PREPARING DATA: {market} ({ticker})")
    print(f"{'='*60}")
    
    # Download data
    df = yf.download(ticker, period="10y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty or len(df) < 500:
        print(f"Not enough data for {market}")
        return None
    
    print(f"Downloaded {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    
    # Calculate features
    df_feat = calculate_features(df)
    feature_cols = get_selected_features(df_feat)
    
    # Create labels based on future returns
    future_returns = df['Close'].pct_change(LABEL_HORIZON).shift(-LABEL_HORIZON)
    
    # Binary labels: 1 = positive return, 0 = negative return
    labels = (future_returns > 0).astype(int)
    
    # Detect trend using HMM
    print("Detecting trends with HMM...")
    is_uptrend = RegimeDetector.detect_hmm(df)
    
    # Prepare sequences
    X_list = []
    y_list = []
    trend_list = []
    
    data = df_feat[feature_cols].values
    
    for i in range(LOOKBACK, len(df) - LABEL_HORIZON):
        sequence = data[i-LOOKBACK:i]
        
        # Normalize sequence
        mean = np.nanmean(sequence, axis=0)
        std = np.nanstd(sequence, axis=0) + 1e-9
        sequence = (sequence - mean) / std
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not np.isnan(labels.iloc[i]) and not np.any(np.isnan(sequence)):
            X_list.append(sequence)
            y_list.append(int(labels.iloc[i]))
            trend_list.append(is_uptrend[i])
    
    X = np.array(X_list)
    y = np.array(y_list)
    trends = np.array(trend_list)
    
    print(f"Created {len(X)} samples")
    print(f"  Uptrend samples: {(trends == 1).sum()}")
    print(f"  Downtrend samples: {(trends == 0).sum()}")
    print(f"  Positive labels: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    print(f"  Negative labels: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    
    return {
        'X': X,
        'y': y,
        'trends': trends,
        'feature_cols': feature_cols
    }


def train_models_for_market(market: str, data: dict):
    """Train and save models for a specific market"""
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS: {market}")
    print(f"{'='*60}")
    
    X = data['X']
    y = data['y']
    trends = data['trends']
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    results = {}
    
    for trend_type in ['uptrend', 'downtrend']:
        print(f"\n--- Training for {trend_type} ---")
        
        # Filter by trend
        if trend_type == 'uptrend':
            mask = trends == 1
        else:
            mask = trends == 0
        
        X_trend = X[mask]
        y_trend = y[mask]
        
        if len(X_trend) < 100:
            print(f"Not enough samples for {trend_type}: {len(X_trend)}")
            continue
        
        print(f"Samples: {len(X_trend)}")
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_trend, y_trend, test_size=0.2, random_state=42, stratify=y_trend
        )
        
        input_shape = (LOOKBACK, X_train.shape[2])
        
        best_model = None
        best_accuracy = 0
        best_model_name = None
        
        # Train each model type
        for model_name, create_fn in [
            ('LSTM', create_lstm_model),
            ('CNN', create_cnn_model),
            ('Transformer', create_transformer_model)
        ]:
            print(f"  Training {model_name}...", end=" ")
            
            try:
                model = create_fn(input_shape)
                
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5)
                ]
                
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
                print(f"Accuracy: {accuracy*100:.1f}%")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = model_name
                    
            except Exception as e:
                print(f"Error: {e}")
        
        # Save best model
        if best_model is not None:
            model_dir = os.path.join(script_dir, f"model_{market}_{trend_type}")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f"{best_model_name}.keras")
            best_model.save(model_path)
            print(f"  Best: {best_model_name} ({best_accuracy*100:.1f}%) saved to {model_path}")
            
            results[trend_type] = {
                'model': best_model_name,
                'accuracy': best_accuracy
            }
    
    return results


def main():
    print("=" * 60)
    print("MARKET-SPECIFIC MODEL TRAINING")
    print(f"Markets: {list(MARKETS_TO_TRAIN.keys())}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    all_results = {}
    
    for market, ticker in MARKETS_TO_TRAIN.items():
        # Prepare data
        data = prepare_data(market, ticker)
        if data is None:
            continue
        
        # Train models
        results = train_models_for_market(market, data)
        all_results[market] = results
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for market, results in all_results.items():
        print(f"\n{market}:")
        for trend, info in results.items():
            print(f"  {trend}: {info['model']} ({info['accuracy']*100:.1f}%)")
    
    print("\nModels saved to market-specific directories.")
    return all_results


if __name__ == "__main__":
    main()
