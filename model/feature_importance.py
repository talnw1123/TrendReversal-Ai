import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Parameters
LOOKBACK = 60

def load_and_process_data(split_name='train'):
    """
    Loads data for a specific split (train/val/test) from the directory structure.
    Returns X (features) and y (labels), and feature_names.
    """
    data_dir = f"trend_data_manual/split/{split_name}"
    files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    
    X_list = []
    y_list = []
    feature_names = []
    
    from features import calculate_features, get_feature_columns
    
    for f in files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        
        # Store label before feature calc
        if 'label' in df.columns:
            labels_series = df['label']
        else:
            continue
            
        # Calculate Features
        df = calculate_features(df)
        if df.empty: continue
        
        # Re-attach label if lost
        if 'label' not in df.columns:
            df['label'] = labels_series
            df = df.dropna()
            
        feature_cols = get_feature_columns(df)
        if not feature_names:
            feature_names = feature_cols
            
        data = df[feature_cols].values
        labels = df['label'].values
        
        if len(data) <= LOOKBACK: continue
        
        # Create Windows
        # For Feature Importance, we can just use the raw rows (no window flattening) 
        # OR we can use the flattened windows. 
        # Using raw rows is simpler and gives importance per feature type (e.g., RSI vs SMA).
        # Using flattened windows gives importance per time-step (e.g., RSI at t-1 vs RSI at t-60).
        # Let's stick to raw feature importance (instantaneous predictive power) first.
        # BUT our model uses windows. 
        # If we use raw rows, we ignore temporal context.
        # Let's use the flattened windows to be consistent with the model input.
        
        for i in range(LOOKBACK, len(data)):
            X_list.append(data[i-LOOKBACK:i].flatten()) # Flatten for RF
            y_list.append(labels[i])
            
    if not X_list:
        return np.array([]), np.array([]), []
        
    return np.array(X_list), np.array(y_list), feature_names

def run_feature_importance():
    print("Loading Training Data...")
    X_train, y_train, feature_names = load_and_process_data('train')
    
    if len(X_train) == 0:
        print("No training data found.")
        return

    print(f"Data Shape: {X_train.shape}")
    
    # We have flattened features: [f1_t-60, f2_t-60, ..., f1_t, f2_t]
    # This is too many features for interpretation if we just want "Is RSI important?".
    # However, RF can handle it.
    # Let's construct names for flattened features.
    
    flat_feature_names = []
    for i in range(LOOKBACK):
        for name in feature_names:
            flat_feature_names.append(f"{name}_t-{LOOKBACK-i}")
            
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    
    # Aggregate importance by base feature name
    # We sum the importance of 'RSI' across all time steps
    feature_importance_dict = {name: 0.0 for name in feature_names}
    
    for i, importance in enumerate(importances):
        # Map index i back to feature name
        # i = time_step * num_features + feature_idx
        feature_idx = i % len(feature_names)
        name = feature_names[feature_idx]
        feature_importance_dict[name] += importance
        
    # Create DataFrame
    importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Normalize so sum is 1 (optional, but good for readability)
    importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
    
    print("\nFeature Importance Ranking:")
    print(importance_df.head(20)) # Print top 20
    
    importance_df.to_csv("feature_importance.csv", index=False)
    print("\nSaved to feature_importance.csv")

    # --- Select Important Features ---
    # Strategy: Select features with importance > threshold OR top K features
    # Let's use a combination: Importance > 0.005 (0.5%)
    
    SELECTED_THRESHOLD = 0.005
    selected_features_df = importance_df[importance_df['Importance'] > SELECTED_THRESHOLD]
    
    # Provide a fallback if too few features are selected
    MIN_FEATURES = 10
    if len(selected_features_df) < MIN_FEATURES:
        print(f"\nWarning: Only {len(selected_features_df)} features met threshold. Selecting top {MIN_FEATURES} instead.")
        selected_features_df = importance_df.head(MIN_FEATURES)
        
    selected_features = selected_features_df['Feature'].tolist()
    
    print(f"\nSelected {len(selected_features)} features.")
    
    # Save to JSON
    import json
    feature_file = "model/selected_features.json"
    
    # Handle running from root or model dir
    if not os.path.exists("model"):
        feature_file = "selected_features.json"
        
    with open(feature_file, "w") as f:
        json.dump(selected_features, f, indent=4)
        
    print(f"Saved selected features to {feature_file}")

if __name__ == "__main__":
    run_feature_importance()
