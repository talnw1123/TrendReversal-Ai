"""
Train separate models for EACH Market and Trend.
Creates folders like: model_BTC_uptrend, model_Gold_downtrend
"""
import pandas as pd
import numpy as np
import os
import glob

# Suppress TF noise BEFORE importing tensorflow
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Reduce retracing warnings
try:
    tf.config.experimental.set_synchronous_execution(True)
except Exception:
    pass

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
BATCH_SIZE = 16 # Reduce batch size for smaller per-market datasets
EPOCHS = 100
LABEL_SMOOTHING = 0.1
BINARY_MODE = True

def get_available_markets():
    """Scan data directory to find all unique market names."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "../trend_data_manual/split/train")
    files = glob.glob(os.path.join(base_dir, "**/*_labeled.csv"), recursive=True)
    
    markets = set()
    for f in files:
        basename = os.path.basename(f)
        # Assuming format: {Market}_{Trend}_labeled.csv
        # e.g. BTC_uptrend_labeled.csv -> BTC
        parts = basename.split('_')
        if len(parts) >= 2:
            markets.add(parts[0])
            
    return sorted(list(markets))

def load_market_trend_data(market, trend_type):
    """Load data for a specific Market AND Trend type."""
    from features import calculate_features, get_selected_features
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for split, X_list, y_list in [('train', X_train, y_train), 
                                   ('val', X_val, y_val), 
                                   ('test', X_test, y_test)]:
        
        base_dir = os.path.join(script_dir, f"../trend_data_manual/split/{split}")
        # Search specifically for this market and trend
        # e.g. *BTC*uptrend*labeled.csv
        pattern = f"*{market}*{trend_type}*_labeled.csv"
        
        files_root = glob.glob(os.path.join(base_dir, pattern))
        files_sub = glob.glob(os.path.join(base_dir, f"**/{pattern}"), recursive=True)
        files = list(set(files_root + files_sub))
        
        for f in files:
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                df = calculate_features(df)
                if df.empty: continue
                
                feature_cols = get_selected_features(df)
                data = df[feature_cols].values
                labels = df['label'].values
                
                if len(data) <= LOOKBACK: continue
                
                for i in range(LOOKBACK, len(data)):
                    seq = data[i-LOOKBACK:i]
                    lbl = labels[i]
                    X_list.append(seq)
                    y_list.append(lbl)
            except Exception as e:
                print(f"Error loading {f}: {e}")

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    if BINARY_MODE:
        def filter_binary(X, y):
            if len(y) == 0: return X, y
            mask = y != 0
            return X[mask], y[mask]
            
        X_train, y_train = filter_binary(X_train, y_train)
        X_val, y_val = filter_binary(X_val, y_val)
        X_test, y_test = filter_binary(X_test, y_test)
        
        # Remap -1 -> 0, 1 -> 1
        if len(y_train) > 0: y_train = np.where(y_train == -1, 0, 1)
        if len(y_val) > 0: y_val = np.where(y_val == -1, 0, 1)
        if len(y_test) > 0: y_test = np.where(y_test == -1, 0, 1)
                
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, model_name, save_dir):
    """Train a single model."""
    from models import build_lstm, build_cnn, build_mlp, build_transformer
    
    # Scale
    scaler = StandardScaler()
    N, T, F = X_train.shape
    scaler.fit(X_train.reshape(-1, F))
    
    X_train = scaler.transform(X_train.reshape(-1, F)).reshape(N, T, F)
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape) if len(X_val) > 0 else X_val
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape) if len(X_test) > 0 else X_test
    
    # Handle NaNs
    X_train = np.nan_to_num(X_train, nan=0.0)
    if len(X_val) > 0: X_val = np.nan_to_num(X_val, nan=0.0)
    if len(X_test) > 0: X_test = np.nan_to_num(X_test, nan=0.0)
    
    num_classes = 2 if BINARY_MODE else 3
    
    # Keras Models
    if model_name in ['LSTM', 'CNN', 'MLP', 'Transformer']:
        if BINARY_MODE:
            y_train_enc = to_categorical(y_train, num_classes=2)
            y_val_enc = to_categorical(y_val, num_classes=2) if len(y_val) > 0 else None
            y_test_enc = to_categorical(y_test, num_classes=2) if len(y_test) > 0 else None
        else:
            y_train_enc = to_categorical(y_train + 1, num_classes=3)
            y_val_enc = to_categorical(y_val + 1, num_classes=3) if len(y_val) > 0 else None
            y_test_enc = to_categorical(y_test + 1, num_classes=3) if len(y_test) > 0 else None
            
        builders = {'LSTM': build_lstm, 'CNN': build_cnn, 'MLP': build_mlp, 'Transformer': build_transformer}
        model = builders[model_name](input_shape=(T, F), num_classes=num_classes)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
        # steps_per_execution=1 prevents intermittent multi_step_on_iterator [64] vs [0] errors
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'],
                      steps_per_execution=1)

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

        val_data = (X_val, y_val_enc) if len(X_val) > 0 else None

        # Adjust batch size if dataset is tiny to avoid empty-batch issues
        effective_bs = min(BATCH_SIZE, max(2, len(X_train) // 8))

        # NEW: Compute class_weight to handle imbalanced classes
        # (fixes MLP 34% issue where it predicts only one class)
        try:
            unique_classes = np.unique(y_train)
            cw = compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, cw)}
        except Exception:
            class_weight_dict = None

        # Try main fit; fall back to simpler config on intermittent TF graph errors
        try:
            model.fit(X_train, y_train_enc,
                      validation_data=val_data, epochs=EPOCHS,
                      batch_size=effective_bs,
                      shuffle=True,
                      class_weight=class_weight_dict,
                      callbacks=callbacks, verbose=0)
        except Exception as e:
            err_msg = str(e)
            if 'Incompatible shapes' in err_msg or 'truediv' in err_msg or 'multi_step' in err_msg:
                print(f"    Warning: TF graph error on {model_name}, retrying with smaller batch...")
                # Rebuild model fresh to clear any partial state
                tf.keras.backend.clear_session()
                model = builders[model_name](input_shape=(T, F), num_classes=num_classes)
                model.compile(loss=loss_fn,
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                              metrics=['accuracy'],
                              steps_per_execution=1, jit_compile=False)
                # Use validation_split instead of validation_data to avoid iterator quirks
                model.fit(X_train, y_train_enc,
                          validation_split=0.0,
                          epochs=min(50, EPOCHS),
                          batch_size=8,
                          shuffle=True,
                          callbacks=callbacks, verbose=0)
            else:
                raise

        acc = 0.0
        if len(X_test) > 0:
            loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)

        filename = os.path.join(save_dir, f"{model_name}.keras")
        model.save(filename)
        return acc

    # ML Models
    else:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1) if len(X_test) > 0 else None
        
        if model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        elif model_name == 'SVM':
            model = SVC(probability=True, class_weight='balanced', random_state=42)
        elif model_name == 'XGBoost':
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1)
            
        model.fit(X_train_flat, y_train)
        
        acc = 0.0
        if X_test_flat is not None:
            y_pred = model.predict(X_test_flat)
            acc = accuracy_score(y_test, y_pred)
            
        filename = os.path.join(save_dir, f"{model_name}.pkl")
        joblib.dump(model, filename)
        return acc

def train_stacking_ensemble(X_train, y_train, X_val, y_val, X_test, y_test,
                            base_models, save_dir):
    """Train a Stacking meta-learner over the trained base models.

    Improved version (v2):
    - Use BOTH train+val predictions (concatenated) to fit meta-learner (more data)
    - Use GradientBoostingClassifier as meta (more powerful than LogReg)
    - Filter out base models with poor val accuracy (<55%) to prevent dragging down

    Returns: test accuracy of the stacked ensemble, or None if not enough base models.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    if len(X_val) == 0 or len(X_test) == 0 or len(X_train) == 0:
        return None

    train_preds = []
    val_preds = []
    test_preds = []
    used_models = []

    for m_name in base_models:
        try:
            kpath = os.path.join(save_dir, f"{m_name}.keras")
            ppath = os.path.join(save_dir, f"{m_name}.pkl")
            if os.path.exists(kpath):
                model = tf.keras.models.load_model(kpath)
                p_tr = model.predict(X_train, verbose=0, batch_size=128)
                p_v = model.predict(X_val, verbose=0, batch_size=128)
                p_t = model.predict(X_test, verbose=0, batch_size=128)
            elif os.path.exists(ppath):
                model = joblib.load(ppath)
                X_tr_flat = X_train.reshape(len(X_train), -1)
                X_v_flat = X_val.reshape(len(X_val), -1)
                X_t_flat = X_test.reshape(len(X_test), -1)
                if hasattr(model, 'predict_proba'):
                    p_tr = model.predict_proba(X_tr_flat)
                    p_v = model.predict_proba(X_v_flat)
                    p_t = model.predict_proba(X_t_flat)
                else:
                    continue
            else:
                continue

            if p_v.shape[1] < 2:
                continue

            # Filter out poor base models (val accuracy < 55%)
            val_pred_class = (p_v[:, 1] >= 0.5).astype(int)
            val_acc = accuracy_score(y_val, val_pred_class)
            if val_acc < 0.55:
                continue

            train_preds.append(p_tr[:, 1])
            val_preds.append(p_v[:, 1])
            test_preds.append(p_t[:, 1])
            used_models.append(m_name)
        except Exception:
            continue

    # Need at least 3 quality base models
    if len(used_models) < 3:
        return None

    train_X = np.column_stack(train_preds)
    val_X = np.column_stack(val_preds)
    test_X = np.column_stack(test_preds)

    # Train on train+val concatenated for more data
    full_X = np.vstack([train_X, val_X])
    full_y = np.concatenate([y_train, y_val])

    # GradientBoosting is more powerful than LR on small features
    meta = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                      learning_rate=0.05, random_state=42)
    meta.fit(full_X, full_y)

    # Evaluate on test
    test_acc = accuracy_score(y_test, meta.predict(test_X))

    # Save
    joblib.dump({
        'meta': meta,
        'base_models': used_models,
        'val_acc_threshold': 0.55,
    }, os.path.join(save_dir, 'Stacking.pkl'))

    return test_acc


def main():
    print("=" * 60)
    print("TRAINING PER-MARKET SPECIALIZED MODELS")
    print("=" * 60)
    
    markets = get_available_markets()
    print(f"Found Markets: {markets}")
    
    results = []
    
    for market in markets:
        for trend in ['uptrend', 'downtrend']:
            print(f"\nProcessing {market} - {trend}...")
            
            X_train, y_train, X_val, y_val, X_test, y_test = load_market_trend_data(market, trend)
            
            if len(X_train) < 50:
                print(f"  -> Not enough training data ({len(X_train)} samples). Skipping.")
                continue
                
            print(f"  Data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            save_dir = f"model_{market}_{trend}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # Train Candidates
            # Restoring full suite of models as requested
            models = ['LSTM', 'CNN', 'MLP', 'Transformer', 'RandomForest', 'SVM', 'XGBoost'] 
            
            best_acc = 0
            best_model = ""
            
            for m_name in models:
                try:
                    acc = train_model(X_train, y_train, X_val, y_val, X_test, y_test, m_name, save_dir)
                    print(f"    {m_name}: {acc:.2%}")

                    if acc > best_acc:
                        best_acc = acc
                        best_model = m_name

                    results.append({
                        'Market': market,
                        'Trend': trend,
                        'Model': m_name,
                        'Accuracy': round(acc * 100, 2),
                        'Samples': len(X_train)
                    })
                except Exception as e:
                    print(f"    Error training {m_name}: {e}")

            # ==================================================================
            # STACKING ENSEMBLE: Train meta-learner on base model predictions
            # ==================================================================
            try:
                stack_acc = train_stacking_ensemble(
                    X_train, y_train, X_val, y_val, X_test, y_test, models, save_dir
                )
                if stack_acc is not None:
                    print(f"    Stacking: {stack_acc:.2%}")
                    if stack_acc > best_acc:
                        best_acc = stack_acc
                        best_model = 'Stacking'
                    results.append({
                        'Market': market, 'Trend': trend,
                        'Model': 'Stacking',
                        'Accuracy': round(stack_acc * 100, 2),
                        'Samples': len(X_train)
                    })
            except Exception as e:
                print(f"    Stacking failed: {e}")

            print(f"  -> Best: {best_model} ({best_acc:.2%})")

    # Save Summary
    if results:
        df = pd.DataFrame(results)
        df.sort_values(by=['Market', 'Trend', 'Accuracy'], ascending=[True, True, False], inplace=True)
        print("\n" + "="*60)
        print(df.to_string(index=False))
        df.to_csv("separate_models_comparison.csv", index=False)
        print("\nSaved to separate_models_comparison.csv")
    else:
        print("No models trained successfully.")

if __name__ == "__main__":
    main()
