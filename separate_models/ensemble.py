import numpy as np
import pandas as pd
import glob
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.utils import to_categorical

# Parameters
LOOKBACK = 60
MODELS_TO_ENSEMBLE = ['LSTM', 'CNN', 'MLP', 'Transformer']

def load_and_process_data(split_name='test'):
    """
    Loads data for a specific split (train/val/test) from the directory structure.
    Returns X (features) and y (labels).
    """
    data_dir = f"trend_data_manual/split/{split_name}"
    # We need to load all files in this directory
    files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    
    X_list = []
    y_list = []
    
    # We need to use the SAME feature calculation as training
    from features import calculate_features, get_feature_columns
    
    for f in files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        
        # Store label before feature calc (which might drop it or return new df)
        if 'label' in df.columns:
            labels_series = df['label']
        else:
            # If label is missing (e.g. in some raw files), skip or handle
            continue
            
        # Calculate Features
        df = calculate_features(df)
        if df.empty: continue
        
        # Re-attach label if lost
        if 'label' not in df.columns:
            df['label'] = labels_series
            # Align indices
            df = df.dropna()
            
        feature_cols = get_feature_columns(df)
        data = df[feature_cols].values
        labels = df['label'].values
        
        if len(data) <= LOOKBACK: continue
        
        # Create Windows
        for i in range(LOOKBACK, len(data)):
            X_list.append(data[i-LOOKBACK:i])
            y_list.append(labels[i])
            
    if not X_list:
        return np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list)

def run_ensemble():
    print("Loading Data...")
    # We need to fit scaler on TRAIN data first
    X_train, _ = load_and_process_data('train')
    scaler = StandardScaler()
    N, T, F = X_train.shape
    scaler.fit(X_train.reshape(-1, F))
    
    # Load Test Data
    X_test, y_test = load_and_process_data('test')
    if len(X_test) == 0:
        print("No test data found.")
        return

    # Scale Test Data
    X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
    
    # Load Models and Predict
    ensemble_probs = np.zeros((len(X_test), 3))
    
    loaded_models = []
    for name in MODELS_TO_ENSEMBLE:
        model_path = f"reversal_model_{name}.keras"
        if not os.path.exists(model_path):
            print(f"Model {name} not found at {model_path}. Skipping.")
            continue
            
        print(f"Loading {name}...")
        model = load_model(model_path)
        loaded_models.append(name)
        
        # Predict
        probs = model.predict(X_test_scaled, verbose=0)
        ensemble_probs += probs
        
    if not loaded_models:
        print("No models loaded.")
        return
        
    # Average Probabilities (Soft Voting)
    ensemble_probs /= len(loaded_models)
    
    # Final Prediction
    y_pred = np.argmax(ensemble_probs, axis=1)
    y_pred_orig = y_pred - 1 # Map back to -1, 0, 1
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred_orig)
    print(f"\nEnsemble ({'+'.join(loaded_models)}) Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_orig, target_names=['Bearish (-1)', 'Neutral (0)', 'Bullish (1)']))
    
    # Save Ensemble Results
    results_df = pd.DataFrame({
        'True_Label': y_test,
        'Pred_Label': y_pred_orig,
        'Prob_Bearish': ensemble_probs[:, 0],
        'Prob_Neutral': ensemble_probs[:, 1],
        'Prob_Bullish': ensemble_probs[:, 2]
    })
    results_df.to_csv("ensemble_predictions.csv", index=False)
    print("Ensemble predictions saved to ensemble_predictions.csv")

if __name__ == "__main__":
    run_ensemble()
