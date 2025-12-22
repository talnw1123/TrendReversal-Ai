"""
Train separate models for Uptrend and Downtrend data.
Creates: model_uptrend_{ModelName}.keras and model_downtrend_{ModelName}.keras
"""
import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from tensorflow.keras.utils import to_categorical
import time

# Parameters
LOOKBACK = 30
BATCH_SIZE = 32
EPOCHS = 100
LABEL_SMOOTHING = 0.1
BINARY_MODE = True  # Set to True to remove Neutral class (High Accuracy Mode)

def load_trend_data(trend_type):
    """Load data for a specific trend type (uptrend or downtrend)."""
    from features import calculate_features, get_selected_features
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    market_test_data = {}
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for split, X_list, y_list in [('train', X_train, y_train), 
                                   ('val', X_val, y_val), 
                                   ('test', X_test, y_test)]:
        base_dir = os.path.join(script_dir, f"../trend_data_manual/split/{split}")
        # Find files in root directory
        files_root = glob.glob(os.path.join(base_dir, f"*_{trend_type}_labeled.csv"))
        # Find files in subdirectories (like train/label/)
        files_sub = glob.glob(os.path.join(base_dir, f"**/*_{trend_type}_labeled.csv"), recursive=True)
        # Combine and remove duplicates
        files = list(set(files_root + files_sub))
        print(f"Found {len(files)} files in {base_dir}")
        for f in files:
            # print(f"Processing {f}...")
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = calculate_features(df)
            if df.empty:
                print(f"  Skipping {f}: Empty after features")
                continue
            
            feature_cols = get_selected_features(df)
            data = df[feature_cols].values
            labels = df['label'].values
            
            if len(data) <= LOOKBACK:
                print(f"  Skipping {f}: Too short ({len(data)} <= {LOOKBACK})")
                continue
            
            # For test set, we want to track which market it belongs to
            market_name = None
            if split == 'test':
                # Filename format: {Market}_{Trend}_labeled.csv
                # e.g. US_uptrend_labeled.csv -> US
                basename = os.path.basename(f)
                market_name = basename.split('_')[0]
                if market_name not in market_test_data:
                    market_test_data[market_name] = ([], [])
            
            for i in range(LOOKBACK, len(data)):
                seq = data[i-LOOKBACK:i]
                lbl = labels[i]
                
                X_list.append(seq)
                y_list.append(lbl)
                
                if split == 'test' and market_name:
                    market_test_data[market_name][0].append(seq)
                    market_test_data[market_name][1].append(lbl)

    # Convert market lists to numpy arrays
    # Convert market lists to numpy arrays
    for m in market_test_data:
        market_test_data[m] = (np.array(market_test_data[m][0]), np.array(market_test_data[m][1]))
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    if BINARY_MODE:
        # Remove Neutral class (0)
        # Keep only -1 (Bearish) and 1 (Bullish)
        print("BINARY MODE: Removing Neutral class...")
        
        def filter_binary(X, y):
            mask = y != 0
            return X[mask], y[mask]
            
        X_train, y_train = filter_binary(X_train, y_train)
        X_val, y_val = filter_binary(X_val, y_val)
        X_test, y_test = filter_binary(X_test, y_test)
        
        # Remap labels: -1 -> 0, 1 -> 1
        y_train = np.where(y_train == -1, 0, 1)
        y_val = np.where(y_val == -1, 0, 1)
        y_test = np.where(y_test == -1, 0, 1)
        
        # Filter markets
        for m in market_test_data:
            X_m, y_m = market_test_data[m]
            mask = y_m != 0
            if len(mask) > 0 and mask.sum() > 0:
                y_m_new = np.where(y_m[mask] == -1, 0, 1)
                market_test_data[m] = (X_m[mask], y_m_new)
            else:
                market_test_data[m] = (np.array([]), np.array([]))
                
    return (X_train, y_train, X_val, y_val, X_test, y_test, market_test_data)

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, market_test_data, model_name, trend_type):
    """Train a single model and save it."""
    from models import build_lstm, build_cnn, build_mlp, build_transformer
    
    # Scale
    scaler = StandardScaler()
    N, T, F = X_train.shape
    scaler.fit(X_train.reshape(-1, F))
    
    X_train = scaler.transform(X_train.reshape(-1, F)).reshape(N, T, F)
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
    
    # Handle any remaining NaN/Inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode labels
    # Encode labels
    num_classes = 2 if BINARY_MODE else 3
    
    if BINARY_MODE:
        # Already 0/1
        y_train_enc = to_categorical(y_train, num_classes=2)
        y_val_enc = to_categorical(y_val, num_classes=2)
        y_test_enc = to_categorical(y_test, num_classes=2)
        y_train_mapped = y_train
    else:
        # -1, 0, 1 -> 0, 1, 2
        y_train_enc = to_categorical(y_train + 1, num_classes=3)
        y_val_enc = to_categorical(y_val + 1, num_classes=3)
        y_test_enc = to_categorical(y_test + 1, num_classes=3)
        y_train_mapped = y_train + 1
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_mapped), y=y_train_mapped)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Build model
    if model_name in ['LSTM', 'CNN', 'MLP', 'Transformer']:
        builders = {
            'LSTM': build_lstm,
            'CNN': build_cnn,
            'MLP': build_mlp,
            'Transformer': build_transformer
        }
        
        model = builders[model_name](input_shape=(T, F), num_classes=num_classes)
        
        # Use label smoothing loss
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
        
        # Use Adam with initial learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
        
        # Callbacks: EarlyStopping + ReduceLROnPlateau
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        # Train
        history = model.fit(
            X_train, y_train_enc,
            validation_data=(X_val, y_val_enc),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate Overall
        loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
        # Evaluate Overall
        loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        if not BINARY_MODE:
            y_pred = y_pred - 1
        
    else:
        # Machine Learning Models (RandomForest, SVM)
        # Flatten data: (N, T, F) -> (N, T*F)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        if model_name == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'SVM':
            model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                random_state=42,
                probability=True  # Needed for ensemble
            )
        elif model_name == 'XGBoost':
            # XGBoost requires labels 0, 1 for binary or 0, 1, 2 for multiclass
            xgb_y_train = y_train if BINARY_MODE else y_train + 1
            
            # Calculate scale_pos_weight for binary
            scale_pos_weight = 1.0
            if BINARY_MODE:
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight if BINARY_MODE else None
            )
            model.fit(X_train_flat, xgb_y_train)
            
            # Evaluate Overall
            y_pred = model.predict(X_test_flat)
            if not BINARY_MODE:
                 y_pred = y_pred - 1 # Map 0,1,2 -> -1,0,1
            
            acc = accuracy_score(y_test, y_pred)
            
            print(f"\n{model_name} ({trend_type}) Overall Accuracy: {acc:.4f}")
            if BINARY_MODE:
                target_names = ['Bearish', 'Bullish']
            else:
                target_names = ['Bearish', 'Neutral', 'Bullish']
            print(classification_report(y_test, y_pred, target_names=target_names))
            
            # Get probabilities
            y_prob = model.predict_proba(X_test_flat)
            
            # Evaluate Per Market
            market_accuracies = {}
            market_probs = {}
            for market, (X_m, y_m) in market_test_data.items():
                if len(X_m) == 0:
                    market_accuracies[f"Acc_{market}"] = 0.0
                    continue
                    
                X_m_scaled = scaler.transform(X_m.reshape(-1, F)).reshape(X_m.shape)
                X_m_scaled = np.nan_to_num(X_m_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                X_m_flat = X_m_scaled.reshape(X_m_scaled.shape[0], -1)
                
                y_m_pred = model.predict(X_m_flat)
                if not BINARY_MODE:
                    y_m_pred = y_m_pred - 1
                    
                m_acc = accuracy_score(y_m, y_m_pred)
                m_prob = model.predict_proba(X_m_flat)
                
                market_accuracies[f"Acc_{market}"] = round(m_acc * 100, 2)
                market_probs[market] = m_prob
                print(f"  {market}: {m_acc*100:.2f}%")
            
            # Save
            save_dir = f"model_{trend_type}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model, filename)
            print(f"Saved to {filename}")
            return acc, market_accuracies, y_prob, market_probs

        if model_name != 'XGBoost':
            model.fit(X_train_flat, y_train) 
            
            # Evaluate Overall
            y_pred = model.predict(X_test_flat)
            acc = accuracy_score(y_test, y_pred)
        
        
    if model_name != 'XGBoost':
        print(f"\n{model_name} ({trend_type}) Overall Accuracy: {acc:.4f}")
    if BINARY_MODE:
        target_names = ['Bearish', 'Bullish']
    else:
        target_names = ['Bearish', 'Neutral', 'Bullish']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Get probabilities for Ensemble
    start_time = time.time()
    if model_name in ['LSTM', 'CNN', 'MLP', 'Transformer']:
        y_prob = model.predict(X_test, verbose=0)
    else:
        y_prob = model.predict_proba(X_test_flat)
    
    # Evaluate Per Market
    market_accuracies = {}
    market_probs = {}
    
    for market, (X_m, y_m) in market_test_data.items():
        if len(X_m) == 0:
            market_accuracies[f"Acc_{market}"] = 0.0
            continue
            
        X_m_scaled = scaler.transform(X_m.reshape(-1, F)).reshape(X_m.shape)
        X_m_scaled = np.nan_to_num(X_m_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        if model_name in ['LSTM', 'CNN', 'MLP', 'Transformer']:
            y_m_enc = to_categorical(y_m, num_classes=num_classes) if BINARY_MODE else to_categorical(y_m + 1, num_classes=3)
            m_loss, m_acc = model.evaluate(X_m_scaled, y_m_enc, verbose=0)
            m_prob = model.predict(X_m_scaled, verbose=0)
        else:
            X_m_flat = X_m_scaled.reshape(X_m_scaled.shape[0], -1)
            y_m_pred = model.predict(X_m_flat)
            m_acc = accuracy_score(y_m, y_m_pred)
            m_prob = model.predict_proba(X_m_flat)
            
        market_accuracies[f"Acc_{market}"] = round(m_acc * 100, 2)
        market_probs[market] = m_prob
        print(f"  {market}: {m_acc*100:.2f}%")

    # Save
    save_dir = f"model_{trend_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if model_name in ['LSTM', 'CNN', 'MLP', 'Transformer']:
        filename = os.path.join(save_dir, f"{model_name}.keras")
        model.save(filename)
    elif model_name != 'XGBoost':
        filename = os.path.join(save_dir, f"{model_name}.pkl")
        joblib.dump(model, filename)
        
    print(f"Saved to {filename}")
    
    return acc, market_accuracies, y_prob, market_probs

def main():
    print("=" * 60)
    print("TRAINING SEPARATE UPTREND & DOWNTREND MODELS")
    print("=" * 60)
    
    results = []
    
    for trend_type in ['uptrend', 'downtrend']:
        print(f"\n{'='*50}")
        print(f"Loading {trend_type.upper()} data...")
        print(f"{'='*50}")
        
        X_train, y_train, X_val, y_val, X_test, y_test, market_test_data = load_trend_data(trend_type)
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        if len(X_train) == 0:
            print(f"No data for {trend_type}")
            continue
        
        # Store predictions for Ensemble
        ensemble_probs = []
        ensemble_market_probs = {} # {market: [prob_model1, prob_model2, ...]}
        
        for model_name in ['LSTM', 'CNN', 'MLP', 'Transformer', 'RandomForest', 'SVM', 'XGBoost']:
            print(f"\n--- Training {model_name} for {trend_type} ---")
            acc, market_accs, y_prob, market_probs = train_model(X_train, y_train, X_val, y_val, X_test, y_test, market_test_data, model_name, trend_type)
            
            # Collect probabilities
            ensemble_probs.append(y_prob)
            for m, prob in market_probs.items():
                if m not in ensemble_market_probs:
                    ensemble_market_probs[m] = []
                ensemble_market_probs[m].append(prob)
            
            res = {
                'Trend': trend_type,
                'Model': model_name,
                'Accuracy': round(acc * 100, 2)
            }
            res.update(market_accs)
            results.append(res)
            
        # ==========================================
        # Calculate Ensemble Result
        # ==========================================
        print(f"\n--- Calculating Ensemble for {trend_type} ---")
        if ensemble_probs:
            # Average probabilities
            avg_probs = np.mean(ensemble_probs, axis=0)
            ensemble_pred = np.argmax(avg_probs, axis=1)
            
            if not BINARY_MODE:
                 # Map back if not binary (0, 1, 2) -> (-1, 0, 1) needed? 
                 # Wait, y_test is already mapped or original?
                 # load_trend_data returns original y (-1, 0, 1)
                 # In train_model we did: y_pred = np.argmax(...) - 1
                 # So here we need strictly -1 if not binary
                 ensemble_pred = ensemble_pred - 1
            
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            print(f"Ensemble ({trend_type}) Overall Accuracy: {ensemble_acc:.4f}")
            
            ens_res = {
                'Trend': trend_type,
                'Model': 'Ensemble',
                'Accuracy': round(ensemble_acc * 100, 2)
            }
            
            # Market Ensemble
            for market, probs_list in ensemble_market_probs.items():
                if not probs_list: continue
                m_avg_probs = np.mean(probs_list, axis=0)
                m_pred = np.argmax(m_avg_probs, axis=1)
                
                # Get true labels for market
                _, y_m_true = market_test_data[market]
                
                if not BINARY_MODE:
                    m_pred = m_pred - 1
                    
                m_acc = accuracy_score(y_m_true, m_pred)
                ens_res[f"Acc_{market}"] = round(m_acc * 100, 2)
                print(f"  {market}: {m_acc*100:.2f}%")
                
            results.append(ens_res)
    
    # Save results
    if not results:
        print("No results to save.")
        return

    results_df = pd.DataFrame(results)
    
    # Reorder columns to put Accuracy first, then Market Accuracies
    cols = ['Trend', 'Model', 'Accuracy'] + [c for c in results_df.columns if c.startswith('Acc_')]
    # Ensure all columns exist (in case some markets were missing in some runs)
    existing_cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[existing_cols]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "separate_models_comparison.csv")
    results_df.to_csv(output_path, index=False)
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
