"""
Evaluate models on each market separately.
"""
import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model
import joblib

# Parameters
LOOKBACK = 60
MARKETS = ['US', 'UK', 'Thai', 'Gold', 'BTC']

def load_market_data(market_name, split_name='test'):
    """Load data for a specific market and split. Uses validation warmup for short datasets."""
    base_dir = "../trend_data_manual/split"
    split_dir = os.path.join(base_dir, split_name)
    
    # Find files for this market
    all_files = glob.glob(os.path.join(split_dir, "**/*_labeled.csv"), recursive=True)
    market_files = [f for f in all_files if market_name in os.path.basename(f)]
    
    from features import calculate_features, get_selected_features
    
    X_list = []
    y_list = []
    
    for f in market_files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        original_len = len(df)
        
        # If test data is too short, prepend validation data for warmup
        if split_name == 'test' and original_len < 300:
            # Find corresponding val file
            val_file = f.replace('/test/', '/val/')
            if os.path.exists(val_file):
                val_df = pd.read_csv(val_file, index_col=0, parse_dates=True)
                df = pd.concat([val_df, df])
                print(f"    Warmup: Prepended {len(val_df)} val rows to {market_name}")
        
        df = calculate_features(df)
        if df.empty:
            continue
        
        feature_cols = get_selected_features(df)
        data = df[feature_cols].values
        labels = df['label'].values
        
        # Only use the test portion (after warmup)
        test_start_idx = max(0, len(data) - (original_len - 120 - LOOKBACK))
        if test_start_idx >= len(data):
            test_start_idx = LOOKBACK  # Fallback
        
        if len(data) <= LOOKBACK:
            continue
        
        for i in range(max(LOOKBACK, test_start_idx), len(data)):
            X_list.append(data[i-LOOKBACK:i])
            y_list.append(labels[i])
    
    return np.array(X_list), np.array(y_list)

def main():
    print("=" * 60)
    print("Per-Market Model Evaluation")
    print("=" * 60)
    
    # Load training data for scaler fitting
    print("\nFitting scaler on training data...")
    X_train_all = []
    
    for market in MARKETS:
        X, _ = load_market_data(market, 'train')
        if len(X) > 0:
            X_train_all.append(X)
    
    X_train_combined = np.vstack(X_train_all)
    N, T, F = X_train_combined.shape
    
    scaler = StandardScaler()
    scaler.fit(X_train_combined.reshape(-1, F))
    
    # Available Models
    keras_models = ['LSTM', 'CNN', 'MLP', 'Transformer']
    ml_models = ['RandomForest', 'SVM']
    
    results = []
    
    for market in MARKETS:
        print(f"\n{'='*40}")
        print(f"Evaluating Market: {market}")
        print(f"{'='*40}")
        
        X_test, y_test = load_market_data(market, 'test')
        
        if len(X_test) == 0:
            print(f"  No test data for {market}")
            continue
        
        # Scale
        X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
        
        for model_name in keras_models + ml_models:
            if model_name in keras_models:
                model_path = f"reversal_model_{model_name}.keras"
                if not os.path.exists(model_path):
                    continue
                model = load_model(model_path)
                probs = model.predict(X_test_scaled, verbose=0)
                y_pred = np.argmax(probs, axis=1) - 1  # Map to -1, 0, 1
            else:
                model_path = f"reversal_model_{model_name}.pkl"
                if not os.path.exists(model_path):
                    continue
                model = joblib.load(model_path)
                X_test_2d = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
                y_pred = model.predict(X_test_2d) - 1  # Map to -1, 0, 1
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Bearish', 'Neutral', 'Bullish'], output_dict=True, zero_division=0)
            
            results.append({
                'Market': market,
                'Model': model_name,
                'Accuracy': round(acc * 100, 2),
                'F1_Macro': round(report['macro avg']['f1-score'], 3),
                'F1_Bearish': round(report['Bearish']['f1-score'], 3),
                'F1_Bullish': round(report['Bullish']['f1-score'], 3)
            })
            
            print(f"  {model_name}: Accuracy={acc*100:.2f}%")
    
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("per_market_results.csv", index=False)
    
    # Print Summary Table
    print("\n" + "=" * 60)
    print("SUMMARY: Best Model per Market")
    print("=" * 60)
    
    for market in MARKETS:
        market_results = results_df[results_df['Market'] == market]
        if len(market_results) > 0:
            best = market_results.loc[market_results['Accuracy'].idxmax()]
            print(f"{market}: {best['Model']} ({best['Accuracy']}%)")
    
    print("\nFull results saved to per_market_results.csv")

if __name__ == "__main__":
    main()
