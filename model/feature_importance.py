import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Parameters
LOOKBACK = 60
OUTPUT_DIR = "model/feature_importance_results"

def process_file_and_get_importance(filepath):
    """
    Loads a single file, processes it, trains a model, and returns feature importance.
    """
    from features import calculate_features, get_feature_columns
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

    # Store label before feature calc
    if 'label' in df.columns:
        labels_series = df['label']
    else:
        return None, None
        
    # Calculate Features
    df = calculate_features(df)
    if df.empty: return None, None
    
    # Re-attach label if lost
    if 'label' not in df.columns:
        df['label'] = labels_series
        df = df.dropna()
        
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        return None, None
        
    data = df[feature_cols].values
    labels = df['label'].values
    
    if len(data) <= LOOKBACK: return None, None
    
    X_list = []
    y_list = []
    
    for i in range(LOOKBACK, len(data)):
        X_list.append(data[i-LOOKBACK:i].flatten())
        y_list.append(labels[i])
        
    if not X_list:
        return None, None
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Train Model
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    
    # Aggregate importance
    feature_importance_dict = {name: 0.0 for name in feature_cols}
    
    for i, importance in enumerate(importances):
        feature_idx = i % len(feature_cols)
        name = feature_cols[feature_idx]
        feature_importance_dict[name] += importance
        
    # Create DataFrame
    importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Normalize
    total_importance = importance_df['Importance'].sum()
    if total_importance > 0:
        importance_df['Importance'] = importance_df['Importance'] / total_importance
        
    return importance_df, feature_cols

def run_feature_importance():
    train_dir = "trend_data_manual/split/train"
    files = glob.glob(os.path.join(train_dir, "**/*.csv"), recursive=True)
    
    if not files:
        print(f"No files found in {train_dir}")
        return

    # Handle running from root or model dir for output path
    output_path = OUTPUT_DIR
    if not os.path.exists("model"):
         # We are likely inside model/ or somewhere else, but let's assume root structure
         # If "model" doesn't exist in CWD, but we are running this script which is in model/..
         # The original script had logic `if not os.path.exists("model"): feature_file = ...`
         # Let's try to be robust.
         if os.path.basename(os.getcwd()) == "model":
             output_path = "feature_importance_results"
    
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Accumulate for global average
    global_importance_sum = {}
    file_count = 0
    
    for f in files:
        # Extract Symbol Name
        # Assuming filename is like .../BTCUSDT_1h.csv or similar
        filename = os.path.basename(f)
        symbol_name = os.path.splitext(filename)[0]
        
        print(f"Processing {symbol_name}...")
        
        importance_df, feature_names = process_file_and_get_importance(f)
        
        if importance_df is not None:
            # Save Individual - Limit to Top 20 for readability
            save_file = os.path.join(output_path, f"feature_importance_{symbol_name}.csv")
            importance_df.head(20).to_csv(save_file, index=False)
            # print(f"  -> Saved to {save_file}")
            
            # Add to global sum
            for _, row in importance_df.iterrows():
                feat = row['Feature']
                imp = row['Importance']
                global_importance_sum[feat] = global_importance_sum.get(feat, 0.0) + imp
            
            file_count += 1
        else:
            print(f"  -> Skipped (Not enough data or error)")
            
    # Calculate Global Average
    if file_count > 0:
        print(f"\nCalculated importance for {file_count} markets.")
        
        global_importance_list = []
        for feat, total_imp in global_importance_sum.items():
            global_importance_list.append({'Feature': feat, 'Importance': total_imp / file_count})
            
        global_df = pd.DataFrame(global_importance_list)
        global_df = global_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        # Save Global - Limit to Top 50
        global_file = "model/feature_importance.csv" if os.path.exists("model") else "feature_importance.csv"
        global_df.head(50).to_csv(global_file, index=False)
        print(f"Saved Global Average to {global_file}")
        
    else:
        print("No valid models trained.")
        
    # --- Robust Feature Selection ---
    select_robust_features(output_path)

def select_robust_features(results_dir):
    print("\n--- Starting Robust Feature Selection (Strict) ---")
    
    files = glob.glob(os.path.join(results_dir, "feature_importance_*.csv"))
    if not files:
        print("No feature importance files found for robust selection.")
        return

    # Parameters for selection
    TOP_K_PER_MARKET = 20   # Stricter: Only look at top 20
    CONSENSUS_THRESHOLD = 0.6  # Stricter: Must be in 60% of markets
    MAX_FINAL_FEATURES = 15 # Hard limit
    
    feature_counts = {}
    total_markets = 0
    
    print(f"Analyzing consensus across {len(files)} markets...")
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            
            # Select Top K for this market
            top_features = df.head(TOP_K_PER_MARKET)['Feature'].tolist()
            
            for feat in top_features:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
                
            total_markets += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if total_markets == 0:
        print("No valid market data to analyze.")
        return
        
    # Filter by consensus
    potential_features = []
    
    # Sort by count (descending)
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nPotential Features (Found in Top {TOP_K_PER_MARKET} of >= {int(total_markets * CONSENSUS_THRESHOLD)} markets):")
    
    for feat, count in sorted_features:
        consistency = count / total_markets
        if consistency >= CONSENSUS_THRESHOLD:
            potential_features.append(feat)
            print(f"  {feat}: {count}/{total_markets} ({consistency:.1%})")
            
    # Final Selection: Take Top MAX_FINAL_FEATURES from potential list
    robust_features = potential_features[:MAX_FINAL_FEATURES]
    
    # Fallback if too few
    MIN_FEATURES = 8
    if len(robust_features) < MIN_FEATURES:
        print(f"\nWarning: Only {len(robust_features)} high-consensus features found. filling up to {MIN_FEATURES} from list.")
        robust_features = [f[0] for f in sorted_features[:MIN_FEATURES]]
        
    print(f"\nFinal Selection: {len(robust_features)} features.")
    print(robust_features)
    
    # Save to JSON
    import json
    feature_file = "model/selected_features.json"
    if not os.path.exists("model"):
        feature_file = "selected_features.json"
        
    with open(feature_file, "w") as f:
        json.dump(robust_features, f, indent=4)
        
    print(f"Saved robust selected features to {feature_file}")

if __name__ == "__main__":
    run_feature_importance()
