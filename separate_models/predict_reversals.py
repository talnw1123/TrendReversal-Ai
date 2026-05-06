import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Parameters (Must match training)
LOOKBACK = 60

def load_and_process_file(file_path):
    """Loads a single CSV, computes features, and returns data + dates."""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Apply Advanced Feature Engineering (Groups A-I)
    from features import calculate_features, get_feature_columns
    df = calculate_features(df)
    
    if df.empty:
        return np.array([]), [], []
        
    feature_cols = get_feature_columns(df)
    data = df[feature_cols].values
    dates = df.index.values
    prices = df['Close'].values
    
    X_list = []
    valid_dates = []
    valid_prices = []
    
    if len(data) <= LOOKBACK:
        return np.array([]), [], []
    
    for i in range(LOOKBACK, len(data)):
        X_list.append(data[i-LOOKBACK:i])
        valid_dates.append(dates[i])
        valid_prices.append(prices[i])
        
    return np.array(X_list), valid_dates, valid_prices

def main():
    print("Loading model...")
    model = load_model("reversal_model.keras")
    
    # We need to fit the scaler on TRAIN data again to be consistent
    # Or load a saved scaler. Since we didn't save it, we must refit.
    print("Refitting scaler on training data...")
    from train_reversal_model import load_and_process_data
    X_train, _ = load_and_process_data('train')
    scaler = StandardScaler()
    N, T, F = X_train.shape
    scaler.fit(X_train.reshape(-1, F))
    
    # Process Test Files
    test_dir = "trend_data_manual/split/test"
    all_files = glob.glob(os.path.join(test_dir, "**/*_labeled.csv"), recursive=True)
    
    results = []
    
    print("Predicting reversals on test data...")
    
    for file_path in all_files:
        market_name = os.path.basename(file_path).split('_')[0] # e.g., US from US_uptrend...
        print(f"  Processing {os.path.basename(file_path)}...")
        
        X_test, dates, prices = load_and_process_file(file_path)
        
        if len(X_test) == 0:
            continue
            
        # Scale
        X_test_flat = X_test.reshape(-1, F)
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
        
        # Predict
        probs = model.predict(X_test_scaled, verbose=0)
        preds = np.argmax(probs, axis=1) # 0=Bearish, 1=Neutral, 2=Bullish
        
        # Identify Reversals (Change in Prediction)
        # We look for transitions TO Bullish (2) or TO Bearish (0)
        
        prev_pred = preds[0]
        
        for i in range(1, len(preds)):
            curr_pred = preds[i]
            date = pd.to_datetime(dates[i])
            price = prices[i]
            
            if curr_pred != prev_pred:
                # URP: Change TO Bullish (2)
                if curr_pred == 2:
                    results.append({
                        'Market': market_name,
                        'Date': date,
                        'Price': price,
                        'Type': 'URP (Upward Reversal)',
                        'Confidence': probs[i][2]
                    })
                # DRP: Change TO Bearish (0)
                elif curr_pred == 0:
                    results.append({
                        'Market': market_name,
                        'Date': date,
                        'Price': price,
                        'Type': 'DRP (Downward Reversal)',
                        'Confidence': probs[i][0]
                    })
            
            prev_pred = curr_pred
            
    # Save Results
    results_df = pd.DataFrame(results)
    output_file = "reversal_points.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nFound {len(results_df)} reversal points.")
    print(f"Saved to {output_file}")
    print(results_df.head())

if __name__ == "__main__":
    main()
