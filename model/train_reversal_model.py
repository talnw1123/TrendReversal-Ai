print("Starting script...")
print("Importing pandas...")
import pandas as pd
import numpy as np
import os
import glob
print("Importing tensorflow...")
import tensorflow as tf
print("Importing sklearn...")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
print("Imports done.")

# Parameters
LOOKBACK = 60
BATCH_SIZE = 32
EPOCHS = 50 # Increased from 20

def load_and_process_data(split_name, market_filter=None):
    """Loads labeled CSVs for a split. Optional: filter by market name."""
    # Use parent directory since script is now in model/
    base_dir = "../trend_data_manual/split"
    split_dir = os.path.join(base_dir, split_name)
    
    # Recursive search to handle subdirectories like 'label'
    all_files = glob.glob(os.path.join(split_dir, "**/*_labeled.csv"), recursive=True)
    
    if market_filter:
        all_files = [f for f in all_files if market_filter in os.path.basename(f)]
    
    X_list = []
    y_list = []
    
    print(f"Loading {split_name} data from {len(all_files)} files...")
    
    from features import calculate_features, get_feature_columns, get_selected_features
    
    for f in all_files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        
        # Calculate Features
        df = calculate_features(df)
        if df.empty: continue
            
        # Use SELECTED features instead of ALL
        feature_cols = get_selected_features(df)
        
        if market_filter and market_filter not in f:
            continue
            
        data = df[feature_cols].values
        labels = df['label'].values
        
        # Create Sliding Windows
        # We need sequences of length LOOKBACK to predict the label at the END (or next step)
        # The label is already aligned with the row. 
        # So X[i] = data[i-LOOKBACK : i], y[i] = label[i]
        
        if len(data) <= LOOKBACK:
            continue
            
        for i in range(LOOKBACK, len(data)):
            X_list.append(data[i-LOOKBACK:i])
            y_list.append(labels[i])
            
    return np.array(X_list), np.array(y_list)

def main():
    # Open output file
    output_file = "model_evaluation_results.txt"
    with open(output_file, "w") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        # 1. Load Data
        X_train, y_train = load_and_process_data('train')
        X_val, y_val = load_and_process_data('val')
        X_test, y_test = load_and_process_data('test')
        
        # 2. Scaling (Fit on Train, Apply to All)
        # X shape is (Samples, Time, Features). We need to flatten to scale, then reshape.
        scaler = StandardScaler()
        
        # Flatten train to (Samples * Time, Features)
        N, T, F = X_train.shape
        X_train_flat = X_train.reshape(-1, F)
        scaler.fit(X_train_flat)
        
        # Transform and Reshape
        X_train = scaler.transform(X_train_flat).reshape(N, T, F)
        X_val = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
        
        # 3. Encode Labels
        # Labels are -1, 0, 1. We need 0, 1, 2 for to_categorical.
        y_train_enc = to_categorical(y_train + 1, num_classes=3)
        y_val_enc = to_categorical(y_val + 1, num_classes=3)
        y_test_enc = to_categorical(y_test + 1, num_classes=3)
        
        log(f"Train shape: {X_train.shape}, {y_train_enc.shape}")
        
        # Calculate Class Weights
        y_train_mapped = y_train + 1
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_mapped),
            y=y_train_mapped
        )
        class_weight_dict = dict(enumerate(class_weights))
        log(f"Class Weights: {class_weight_dict}")
        
        # 4. Train and Evaluate Multiple Models
        from models import build_lstm, build_cnn, build_mlp, build_transformer
        # from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        import joblib
        
        # Define models: (Builder Function OR Class, Is_Keras)
        models_dict = {
            'LSTM': (build_lstm, True),
            'CNN': (build_cnn, True),
            'MLP': (build_mlp, True),
            'Transformer': (build_transformer, True),
            # 'XGBoost': (XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1), False),
            'RandomForest': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'), False),
            'SVM': (SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42), False)
        }
        
        comparison_results = []
        
        for name, (model_def, is_keras) in models_dict.items():
            log(f"\n{'='*30}")
            log(f"Training {name} Model...")
            log(f"{'='*30}")
            
            if is_keras:
                model = model_def(input_shape=(T, F), num_classes=3)
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                
                # Train Keras
                history = model.fit(
                    X_train, y_train_enc,
                    validation_data=(X_val, y_val_enc),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    class_weight=class_weight_dict,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                    verbose=1
                )
                
                # Evaluate Global
                loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
                y_pred_probs = model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_pred_probs, axis=1)
                
                # Save Keras Model
                model_filename = f"reversal_model_{name}.keras"
                model.save(model_filename)
                
            else:
                # ML Models (Sklearn/XGB) need 2D input (Samples, Time*Features)
                # We already have X_train_flat from scaling step, but we need to reshape X_train (which is 3D) back to 2D
                # Actually, X_train was reshaped to 3D after scaling.
                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                X_val_2d = X_val.reshape(X_val.shape[0], -1)
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                
                # Labels: ML models take 1D labels (0, 1, 2), not One-Hot
                # y_train is -1, 0, 1. Map to 0, 1, 2
                y_train_ml = y_train + 1
                y_val_ml = y_val + 1
                y_test_ml = y_test + 1
                
                model = model_def
                model.fit(X_train_2d, y_train_ml)
                
                # Evaluate
                acc = model.score(X_test_2d, y_test_ml)
                if hasattr(model, "predict_proba"):
                    y_pred_probs = model.predict_proba(X_test_2d)
                else:
                    # SVM needs probability=True
                    y_pred_probs = model.decision_function(X_test_2d) # Not probs, but ok for now if predict_proba fails
                    
                y_pred = model.predict(X_test_2d)
                
                # Save ML Model
                model_filename = f"reversal_model_{name}.pkl"
                joblib.dump(model, model_filename)
            
            log(f"\n{name} Global Test Accuracy: {acc:.4f}")
            log(f"Saved {name} model to {model_filename}")
            
            # Detailed Metrics
            # Map predictions back to -1, 0, 1
            # If Keras: y_pred is 0,1,2. y_test_orig is -1,0,1.
            # If ML: y_pred is 0,1,2 (mapped from input).
            
            # Wait, for ML models: y_train_ml was mapped +1. So y_pred will be 0,1,2.
            # We need to subtract 1 to get back to -1,0,1
            y_pred_orig = y_pred - 1
            y_test_orig = y_test
            
            report = classification_report(y_test_orig, y_pred_orig, target_names=['Bearish (-1)', 'Neutral (0)', 'Bullish (1)'], output_dict=True)
            log(classification_report(y_test_orig, y_pred_orig, target_names=['Bearish (-1)', 'Neutral (0)', 'Bullish (1)']))
            
            # Collect Metrics for Comparison
            comparison_results.append({
                'Model': name,
                'Accuracy': acc,
                'F1_Macro': report['macro avg']['f1-score'],
                'F1_Weighted': report['weighted avg']['f1-score'],
                'F1_Bearish': report['Bearish (-1)']['f1-score'],
                'F1_Bullish': report['Bullish (1)']['f1-score']
            })
            
            # Update default model if it's the best (optional logic, skipping for now)
        
        # Save Comparison CSV
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv("model_comparison.csv", index=False)
        log("\nModel Comparison Summary:")
        log(comparison_df.to_string())
        print("\nComparison saved to model_comparison.csv")

if __name__ == "__main__":
    main()
