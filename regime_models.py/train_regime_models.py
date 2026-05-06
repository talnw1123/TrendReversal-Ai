"""
Train Regime Detection Models
Trains and saves Random Forest, GMM, and HMM models for each market as .pkl files.
These models can be loaded by RegimeDetector for faster inference.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.decomposition.asf import ASF

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../all_trad'))

from model.features import calculate_features
from model.regime_detection import RegimeDetector
from all_trad.compare_regime_methods import calculate_trend_returns

# Market tickers
TICKERS = {
    'Thai': '^SET.BK',
    'UK': '^FTSE',
    'Gold': 'GC=F',
    'US': '^GSPC',
    'BTC': 'BTC-USD'
}

# Output directory for models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'regime_models')


def train_random_forest(df_feat, market_name, valid_features):
    """Train and save Random Forest model"""
    print("  [RF] Training Random Forest...")
    
    # Create Target
    df_feat['future_ret'] = df_feat['Close'].shift(-20) / df_feat['Close'] - 1
    df_feat['target'] = (df_feat['future_ret'] > 0).astype(int)
    
    valid_mask = ~df_feat['future_ret'].isna()
    X = df_feat.loc[valid_mask, valid_features].values
    y = df_feat.loc[valid_mask, 'target'].values
    X = np.nan_to_num(X)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Save
    model_path = os.path.join(MODEL_DIR, f'regime_rf_{market_name}.pkl')
    features_path = os.path.join(MODEL_DIR, f'regime_features_{market_name}.pkl')
    
    joblib.dump(rf, model_path)
    joblib.dump(valid_features, features_path)
    
    print(f"  [RF] ✅ Saved: {model_path}")
    return rf


def train_xgboost(df_feat, market_name, valid_features):
    """Train and save XGBoost model"""
    print("  [XGB] Training XGBoost...")
    
    # Create Target
    df_feat['future_ret'] = df_feat['Close'].shift(-20) / df_feat['Close'] - 1
    df_feat['target'] = (df_feat['future_ret'] > 0).astype(int)
    
    valid_mask = ~df_feat['future_ret'].isna()
    X = df_feat.loc[valid_mask, valid_features].values
    y = df_feat.loc[valid_mask, 'target'].values
    X = np.nan_to_num(X)
    
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss', use_label_encoder=False)
    xgb.fit(X, y)
    
    # Save
    model_path = os.path.join(MODEL_DIR, f'regime_xgb_{market_name}.pkl')
    features_path = os.path.join(MODEL_DIR, f'regime_features_{market_name}.pkl') # Reuses same feature file logic
    
    joblib.dump(xgb, model_path)
    
    print(f"  [XGB] ✅ Saved: {model_path}")
    return xgb


def train_svc(df_feat, market_name, valid_features):
    """Train and save SVC model"""
    print("  [SVC] Training SVC...")
    
    # Create Target
    df_feat['future_ret'] = df_feat['Close'].shift(-20) / df_feat['Close'] - 1
    df_feat['target'] = (df_feat['future_ret'] > 0).astype(int)
    
    valid_mask = ~df_feat['future_ret'].isna()
    X = df_feat.loc[valid_mask, valid_features].values
    y = df_feat.loc[valid_mask, 'target'].values
    X = np.nan_to_num(X)
    
    # Need scaling for SVC
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    svc.fit(X_scaled, y)
    
    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, f'regime_svc_{market_name}.pkl')
    scaler_path = os.path.join(MODEL_DIR, f'regime_svc_scaler_{market_name}.pkl')
    
    joblib.dump(svc, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  [SVC] ✅ Saved: {model_path}")
    return svc


def train_logistic_regression(df_feat, market_name, valid_features):
    """Train and save Logistic Regression model"""
    print("  [LR] Training Logistic Regression...")
    
    # Create Target
    df_feat['future_ret'] = df_feat['Close'].shift(-20) / df_feat['Close'] - 1
    df_feat['target'] = (df_feat['future_ret'] > 0).astype(int)
    
    valid_mask = ~df_feat['future_ret'].isna()
    X = df_feat.loc[valid_mask, valid_features].values
    y = df_feat.loc[valid_mask, 'target'].values
    X = np.nan_to_num(X)
    
    # Need scaling for LR typically
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    
    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, f'regime_lr_{market_name}.pkl')
    scaler_path = os.path.join(MODEL_DIR, f'regime_lr_scaler_{market_name}.pkl')
    
    joblib.dump(lr, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  [LR] ✅ Saved: {model_path}")
    return lr


def train_gmm(df, market_name, window=20):
    """Train and save GMM model"""
    print("  [GMM] Training Gaussian Mixture Model...")
    
    data = df.copy()
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['volatility'] = data['log_ret'].rolling(window).std()
    data = data.dropna()
    
    if len(data) < 100:
        print("  [GMM] Not enough data")
        return None
    
    X = data[['log_ret', 'volatility']].values
    
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=10)
    gmm.fit(X)
    
    # Determine which component is uptrend (higher mean return)
    means = gmm.means_[:, 0]
    uptrend_label = np.argmax(means)
    
    # Save model and metadata
    model_path = os.path.join(MODEL_DIR, f'regime_gmm_{market_name}.pkl')
    meta_path = os.path.join(MODEL_DIR, f'regime_gmm_meta_{market_name}.pkl')
    
    joblib.dump(gmm, model_path)
    joblib.dump({'uptrend_label': uptrend_label, 'window': window}, meta_path)
    
    print(f"  [GMM] ✅ Saved: {model_path}")
    return gmm


def train_hmm(df, market_name):
    """Train and save HMM model"""
    print("  [HMM] Training Hidden Markov Model...")
    
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [HMM] ⚠️ hmmlearn not installed. Skipping HMM.")
        return None
    
    data = df.copy()
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    valid_data = data.dropna().copy()
    
    if len(valid_data) < 100:
        print("  [HMM] Not enough data")
        return None
    
    X = valid_data['log_ret'].values.reshape(-1, 1)
    
    hmm = GaussianHMM(n_components=2, covariance_type='diag', n_iter=100, random_state=42)
    try:
        hmm.fit(X)
    except Exception as e:
        print(f"  [HMM] Training error: {e}")
        return None
    
    # Determine which state is uptrend
    means = hmm.means_.flatten()
    uptrend_state = np.argmax(means)
    
    # Save
    model_path = os.path.join(MODEL_DIR, f'regime_hmm_{market_name}.pkl')
    meta_path = os.path.join(MODEL_DIR, f'regime_hmm_meta_{market_name}.pkl')
    
    joblib.dump(hmm, model_path)
    joblib.dump({'uptrend_state': uptrend_state}, meta_path)
    
    print(f"  [HMM] ✅ Saved: {model_path}")
    return hmm


def train_all_models(ticker, market_name):
    """Train all regime models for a market"""
    print(f"\n{'='*60}")
    print(f"Training Models for {market_name} ({ticker})")
    print("=" * 60)
    
    # 1. Fetch Data
    try:
        df = yf.download(ticker, period="max", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty or len(df) < 500:
        print("Not enough data.")
        return

    print(f"  Data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    
    # 2. Calculate Features for RF
    print("  Calculating features...")
    df_feat = calculate_features(df)
    
    # Load feature list
    csv_path = os.path.join(os.path.dirname(__file__), f'regime_features_{market_name}.csv')
    if os.path.exists(csv_path):
        try:
            feature_df = pd.read_csv(csv_path)
            selected_features = feature_df['Feature'].tolist()
        except:
            selected_features = []
    else:
        selected_features = [
            'volume_nvi', 'volume_obv', 'vma_120', 'volume_vpt', 'ma_120', 'vma_60',
            'p_ma_dist_60_120', 'trend_visual_ichimoku_b', 'vma_20', 'volume_adi', 'ma_60'
        ]
    
    valid_features = [f for f in selected_features if f in df_feat.columns]
    
    # Create output directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 3. Train Models
    methods_to_evaluate = ['SMA200', 'GMM', 'HMM', 'ADX_Supertrend']
    if valid_features:
        train_random_forest(df_feat, market_name, valid_features)
        train_xgboost(df_feat, market_name, valid_features)
        train_svc(df_feat, market_name, valid_features)
        train_logistic_regression(df_feat, market_name, valid_features)
        methods_to_evaluate.extend(['RandomForest', 'XGBoost', 'SVC', 'LogisticRegression'])
    else:
        print("  [RF/XGB/SVC/LR] ⚠️ No valid features, skipping supervised models")
    
    train_gmm(df, market_name)
    train_hmm(df, market_name)
    
    # 4. Evaluate and Select Best Model via Pareto + ASF
    print(f"\\n  [MOO] Evaluating models for {market_name} (Out-of-Sample) to find the best...")
    
    # Define OOS Split (Train 80%, Test 20%)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    # Pre-compute features for the split if needed for supervised models
    df_feat_train = df_feat.iloc[:split_idx].copy() if valid_features else None
    df_feat_test = df_feat.iloc[split_idx:].copy() if valid_features else None
    
    # Define how to get OOS regime for each method
    def get_oos_regime(method, _df_test, _market_name):
        if method == 'SMA200':
            # Uses standard logic on test set
            return (_df_test['Close'] > _df_test['Close'].rolling(200).mean()).astype(int).fillna(0).values
        elif method == 'ADX_Supertrend':
            return RegimeDetector.detect_adx_supertrend(_df_test)
        elif method == 'GMM':
            return RegimeDetector.detect_gmm(_df_test, market_name=_market_name)
        elif method == 'HMM':
            return RegimeDetector.detect_hmm(_df_test, market_name=_market_name)
        elif method == 'RandomForest':
            # Temporary OOS training to prevent lookahead bias
            if not valid_features or len(df_feat_train) < 100: return np.ones(len(_df_test))
            df_feat_train['future_ret'] = df_feat_train['Close'].shift(-20) / df_feat_train['Close'] - 1
            df_feat_train['target'] = (df_feat_train['future_ret'] > 0).astype(int)
            mask = ~df_feat_train['future_ret'].isna()
            X_tr = np.nan_to_num(df_feat_train.loc[mask, valid_features].values)
            y_tr = df_feat_train.loc[mask, 'target'].values
            if len(X_tr) < 50: return np.ones(len(_df_test))
            
            from sklearn.ensemble import RandomForestClassifier
            tmp_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            tmp_rf.fit(X_tr, y_tr)
            
            X_te = np.nan_to_num(df_feat_test[valid_features].values)
            return tmp_rf.predict(X_te)
            
        elif method == 'XGBoost':
            if not valid_features or len(df_feat_train) < 100: return np.ones(len(_df_test))
            df_feat_train['future_ret'] = df_feat_train['Close'].shift(-20) / df_feat_train['Close'] - 1
            df_feat_train['target'] = (df_feat_train['future_ret'] > 0).astype(int)
            mask = ~df_feat_train['future_ret'].isna()
            X_tr = np.nan_to_num(df_feat_train.loc[mask, valid_features].values)
            y_tr = df_feat_train.loc[mask, 'target'].values
            if len(X_tr) < 50: return np.ones(len(_df_test))
            
            from xgboost import XGBClassifier
            tmp_xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss', use_label_encoder=False)
            tmp_xgb.fit(X_tr, y_tr)
            
            X_te = np.nan_to_num(df_feat_test[valid_features].values)
            return tmp_xgb.predict(X_te)
            
        elif method == 'SVC':
            if not valid_features or len(df_feat_train) < 100: return np.ones(len(_df_test))
            df_feat_train['future_ret'] = df_feat_train['Close'].shift(-20) / df_feat_train['Close'] - 1
            df_feat_train['target'] = (df_feat_train['future_ret'] > 0).astype(int)
            mask = ~df_feat_train['future_ret'].isna()
            X_tr = np.nan_to_num(df_feat_train.loc[mask, valid_features].values)
            y_tr = df_feat_train.loc[mask, 'target'].values
            if len(X_tr) < 50: return np.ones(len(_df_test))
            
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            tmp_scaler = StandardScaler()
            X_tr_sc = tmp_scaler.fit_transform(X_tr)
            
            tmp_svc = SVC(kernel='rbf', probability=True, random_state=42)
            tmp_svc.fit(X_tr_sc, y_tr)
            
            X_te_sc = tmp_scaler.transform(np.nan_to_num(df_feat_test[valid_features].values))
            return tmp_svc.predict(X_te_sc)
            
        elif method == 'LogisticRegression':
            if not valid_features or len(df_feat_train) < 100: return np.ones(len(_df_test))
            df_feat_train['future_ret'] = df_feat_train['Close'].shift(-20) / df_feat_train['Close'] - 1
            df_feat_train['target'] = (df_feat_train['future_ret'] > 0).astype(int)
            mask = ~df_feat_train['future_ret'].isna()
            X_tr = np.nan_to_num(df_feat_train.loc[mask, valid_features].values)
            y_tr = df_feat_train.loc[mask, 'target'].values
            if len(X_tr) < 50: return np.ones(len(_df_test))
            
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            tmp_scaler = StandardScaler()
            X_tr_sc = tmp_scaler.fit_transform(X_tr)
            
            tmp_lr = LogisticRegression(max_iter=1000, random_state=42)
            tmp_lr.fit(X_tr_sc, y_tr)
            
            X_te_sc = tmp_scaler.transform(np.nan_to_num(df_feat_test[valid_features].values))
            return tmp_lr.predict(X_te_sc)
            
        return np.ones(len(_df_test)) # fallback
        
    model_metrics = []
    
    for method in methods_to_evaluate:
        try:
            regime_signals = get_oos_regime(method, df_test, market_name)
            metrics = calculate_trend_returns(df_test, regime_signals)
            
            # We want to maximize Return, Separation, Sharpe, and minimize Max DD. 
            # We skip if excess_return or return is extremely bad to avoid NaN scaling issues.
            if pd.isna(metrics['sharpe']) or pd.isna(metrics['max_dd']):
                continue
                
            model_metrics.append({
                'Method': method,
                'Return': metrics['strategy_return'],
                'Separation': metrics['separation'],
                'Sharpe': metrics['sharpe'],
                'MaxDD': metrics['max_dd'] # Negative number
            })
            print(f"    - {method:20}: Ret={metrics['strategy_return']:+6.1f}% | Sep={metrics['separation']:+6.3f} | Sharpe={metrics['sharpe']:4.2f} | MaxDD={metrics['max_dd']:6.1f}%")
        except Exception as e:
            print(f"    - {method:20}: Failed evaluation ({e})")
            
    if not model_metrics:
        print("  [MOO] ⚠️ No models evaluated successfully. Defaulting to SMA200.")
        best_model = 'SMA200'
    else:
        # Pareto front calculation
        # Objectives formulated as minimization
        # 1. Minimize (-Return)
        # 2. Minimize (-Separation)
        # 3. Minimize (-Sharpe)
        # 4. Minimize MaxDD (since max_dd is usually negative, we want to bring it closer to 0, so minimize absolute value? Or just maximize MaxDD as it is negative e.g. -10 is better than -50. So minimize (-MaxDD))
        
        n_models = len(model_metrics)
        F = np.zeros((n_models, 4))
        for i, m in enumerate(model_metrics):
            F[i, 0] = -m['Return']
            F[i, 1] = -m['Separation']
            F[i, 2] = -m['Sharpe']
            F[i, 3] = -m['MaxDD'] # Maximize negative drawdown
            
        fronts = NonDominatedSorting().do(F)
        pareto_front_indices = fronts[0]
        
        print(f"  [MOO] Pareto Front size: {len(pareto_front_indices)}/{n_models}")
        
        # ASF to select best from Pareto front
        F_pareto = F[pareto_front_indices]
        decomp = ASF()
        weights = np.array([0.4, 0.2, 0.2, 0.2]) # Weight Return highest, then others
        
        # Normalize
        F_range = F_pareto.max(axis=0) - F_pareto.min(axis=0)
        F_range[F_range == 0] = 1e-6
        F_norm = (F_pareto - F_pareto.min(axis=0)) / F_range
        
        asf_values = decomp.do(F_norm, 1 / weights)
        best_pareto_idx = asf_values.argmin()
        best_global_idx = pareto_front_indices[best_pareto_idx]
        
        best_model = model_metrics[best_global_idx]['Method']
        print(f"  [MOO] 🔥 Best Model selected by ASF: {best_model} 🔥")
        
    # Save best model selection
    meta_path = os.path.join(MODEL_DIR, f'best_regime_meta_{market_name}.pkl')
    joblib.dump({'best_model_type': best_model}, meta_path)
    print(f"  [MOO] ✅ Saved best model selection to: {meta_path}")

    # Return the metrics for aggregation
    for m in model_metrics:
        m['Market'] = market_name
        m['Is_Best'] = (m['Method'] == best_model)
    return model_metrics


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING ALL REGIME DETECTION MODELS (RF, XGB, SVC, LR, GMM, HMM)")
    print("=" * 60)
    
    all_results = []
    
    for market, ticker in TICKERS.items():
        market_metrics = train_all_models(ticker, market)
        if market_metrics:
            all_results.extend(market_metrics)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        # Reorder columns
        cols = ['Market', 'Method', 'Is_Best', 'Return', 'Separation', 'Sharpe', 'MaxDD']
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
        
        csv_path = os.path.join(MODEL_DIR, 'regime_evaluation_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"\\n✅ Saved comprehensive model evaluation summary to: {csv_path}")

    print("\\n" + "=" * 60)
    print("ALL FORMATS TRAINED AND SAVED")
    print(f"Output directory: {MODEL_DIR}")
    print("=" * 60)
