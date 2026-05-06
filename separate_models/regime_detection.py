import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

class RegimeDetector:
    """
    Detects market regimes (Uptrend/Downtrend) using advanced methods:
    1. GMM (Gaussian Mixture Model) on Returns & Volatility.
    2. ADX + Supertrend (Trend Strength).
    3. HMM (Hidden Markov Model) on Returns.
    """
    
    @staticmethod
    def detect_gmm(df, window=20, market_name='Default'):
        """
        Uses GMM to cluster market into 2 states based on Log Returns and Volatility.
        Tries to load pre-trained model from .pkl first.
        """
        import os
        import joblib
        
        data = df.copy()
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['log_ret'].rolling(window=window).std()
        valid_data = data.dropna().copy()
        
        if len(valid_data) < 100:
            return np.ones(len(df))
        
        X = valid_data[['log_ret', 'volatility']].values
        
        # Try to load pre-trained model
        model_dir = os.path.join(os.path.dirname(__file__), '../all_trad/regime_models')
        model_path = os.path.join(model_dir, f'regime_gmm_{market_name}.pkl')
        meta_path = os.path.join(model_dir, f'regime_gmm_meta_{market_name}.pkl')
        
        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                gmm = joblib.load(model_path)
                meta = joblib.load(meta_path)
                uptrend_label = meta['uptrend_label']
                
                labels = gmm.predict(X)
                regime = np.where(labels == uptrend_label, 1, 0)
                
                full_regime = pd.Series(0, index=df.index)
                full_regime.loc[valid_data.index] = regime
                return full_regime.values
                
            except Exception as e:
                print(f"  [GMM Warning] Error loading model: {e}")
        
        # Fallback: Train on-the-fly
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)
            labels = gmm.predict(X)
        except Exception as e:
            sma = data['Close'].rolling(200).mean()
            return (data['Close'] > sma).astype(int).fillna(0).values
        
        mean_ret_0 = X[labels == 0, 0].mean()
        mean_ret_1 = X[labels == 1, 0].mean()
        uptrend_label = 0 if mean_ret_0 > mean_ret_1 else 1
        
        regime = np.where(labels == uptrend_label, 1, 0)
        
        full_regime = pd.Series(0, index=df.index)
        full_regime.loc[valid_data.index] = regime
        
        return full_regime.values

    @staticmethod
    def detect_adx_supertrend(df, adx_period=14, adx_threshold=25, atr_period=10, multiplier=3.0):
        """
        Uptrend if ADX > 25 AND Supertrend is Green (Close > Supertrend).
        Downtrend if ADX > 25 AND Supertrend is Red.
        Sideways (0) if ADX < 25. -> We treat Sideways as Downtrend (Cash) for safety, or separate state?
        For binary 'is_uptrend', we only return 1 if Strong Uptrend.
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate ADX
        adx_ind = ADXIndicator(high, low, close, window=adx_period)
        adx = adx_ind.adx()
        
        # Calculate Supertrend
        # Manual Supertrend calculation as 'ta' library might not have it directly or easy to access
        atr = AverageTrueRange(high, low, close, window=atr_period).average_true_range()
        
        hl2 = (high + low) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        final_upper = pd.Series(0.0, index=df.index)
        final_lower = pd.Series(0.0, index=df.index)
        supertrend = pd.Series(0.0, index=df.index) # 1 = Green (Up), -1 = Red (Down)
        
        # Iterative calculation needed for Supertrend state
        # Optimized loop or use pandas logic where possible
        # For simplicity/speed in backtest loop:
        # This is O(N), acceptable.
        
        # Initialize
        final_upper.iloc[0] = basic_upper.iloc[0]
        final_lower.iloc[0] = basic_lower.iloc[0]
        
        for i in range(1, len(df)):
            # Upper Band
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
                
            # Lower Band
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
                
            # Supertrend
            prev_st = supertrend.iloc[i-1]
            if prev_st == 1:
                if close.iloc[i] <= final_lower.iloc[i]:
                    supertrend.iloc[i] = -1
                else:
                    supertrend.iloc[i] = 1
            elif prev_st == -1:
                if close.iloc[i] >= final_upper.iloc[i]:
                    supertrend.iloc[i] = 1
                else:
                    supertrend.iloc[i] = -1
            else: # Start case
                if close.iloc[i] > close.iloc[i-1]:
                     supertrend.iloc[i] = 1
                else:
                    supertrend.iloc[i] = -1

        # Logic: Uptrend (1) only if ADX > threshold and Supertrend == 1
        is_uptrend = ((adx > adx_threshold) & (supertrend == 1)).astype(int)
        
        return is_uptrend.values

    @staticmethod
    def detect_hmm(df, market_name='Default'):
        """
        Uses Hidden Markov Model to infer hidden states (Bull/Bear) from Returns.
        Tries to load pre-trained model from .pkl first.
        """
        import os
        import joblib
        
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            import sys
            import subprocess
            print("  [Warning] hmmlearn not found. Auto-installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn"])
            from hmmlearn.hmm import GaussianHMM

        data = df.copy()
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        valid_data = data.dropna().copy()
        
        if len(valid_data) < 100:
             return np.ones(len(df))
             
        X = valid_data[['log_ret']].values
        
        # Try to load pre-trained model
        model_dir = os.path.join(os.path.dirname(__file__), '../all_trad/regime_models')
        model_path = os.path.join(model_dir, f'regime_hmm_{market_name}.pkl')
        meta_path = os.path.join(model_dir, f'regime_hmm_meta_{market_name}.pkl')
        
        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                model = joblib.load(model_path)
                meta = joblib.load(meta_path)
                uptrend_state = meta['uptrend_state']
                
                hidden_states = model.predict(X)
                regime = np.where(hidden_states == uptrend_state, 1, 0)
                
                full_regime = pd.Series(0, index=df.index)
                full_regime.loc[valid_data.index] = regime
                return full_regime.values
                
            except Exception as e:
                print(f"  [HMM Warning] Error loading model: {e}")
        
        # Fallback: Train on-the-fly
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        try:
            model.fit(X)
        except:
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values
            
        hidden_states = model.predict(X)
        
        means = model.means_[:, 0]
        bull_state = np.argmax(means)
        
        regime = np.where(hidden_states == bull_state, 1, 0)
        
        full_regime = pd.Series(0, index=df.index)
        full_regime.loc[valid_data.index] = regime
        
        return full_regime.values

    @staticmethod
    def detect_gmm_enhanced(df, window=20):
        """
        [ENHANCED] GMM with 3 States + More Features + Scaling
        Features: Log Return, Volatility, RSI, ADX
        """
        from sklearn.preprocessing import StandardScaler
        from ta.momentum import RSIIndicator
        
        data = df.copy()
        
        # 1. Feature Engineering
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['log_ret'].rolling(window=window).std()
        
        # RSI
        rsi_ind = RSIIndicator(data['Close'], window=14)
        data['rsi'] = rsi_ind.rsi()
        
        # ADX
        adx_ind = ADXIndicator(data['High'], data['Low'], data['Close'], window=14)
        data['adx'] = adx_ind.adx()
        
        # Drop NaNs
        valid_data = data.dropna().copy()
        
        if len(valid_data) < 200:
             return np.ones(len(df)) # Fallback
             
        # Select Features
        X = valid_data[['log_ret', 'volatility', 'rsi', 'adx']].values
        
        # 2. Scaling (Critical for Distance-based models like GMM)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            # 3. GMM with 3 Components (Bull, Bear, Sideways)
            gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
            gmm.fit(X_scaled)
            labels = gmm.predict(X_scaled)
            
            # 4. Identify Uptrend State
            # We look for the cluster with the Highest Mean Return (from original data, not scaled if possible, but identifying via labels)
            # Or use the Mean of 'log_ret' column for each cluster index
            
            cluster_means = []
            for i in range(3):
                # Calculate mean return for this cluster
                mask = (labels == i)
                if np.sum(mask) > 0:
                    avg_ret = valid_data.loc[valid_data.index[mask], 'log_ret'].mean()
                    cluster_means.append(avg_ret)
                else:
                    cluster_means.append(-999) # Empty cluster
            
            uptrend_state = np.argmax(cluster_means)
            
            # Map to binary (1 = Uptrend, 0 = Others)
            regime = np.where(labels == uptrend_state, 1, 0)
            
            # Realign
            full_regime = pd.Series(0, index=df.index)
            full_regime.loc[valid_data.index] = regime
            
            return full_regime.values
            
        except Exception as e:
            print(f"  [GMM Enhanced Error] {e}. Falling back to SMA200.")
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values

    @staticmethod
    def detect_hmm_enhanced(df):
        """
        [ENHANCED] HMM with 3 States + Volatility Feature
        Features: Log Return, Volatility
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            import sys
            import subprocess
            print("  [Warning] hmmlearn not found. Auto-installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn"])
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
        data = df.copy()
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        # Add Volatility (helps distinguish high vol bear vs low vol bull)
        data['volatility'] = data['log_ret'].rolling(window=20).std()
        
        valid_data = data.dropna().copy()
        
        if len(valid_data) < 200:
             return np.ones(len(df))
             
        X = valid_data[['log_ret', 'volatility']].values
        
        # Scaling is less critical for HMM than GMM but can help convergence with mixed features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            # 3 States: Bull, Bear, Sideways
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            model.fit(X_scaled)
            hidden_states = model.predict(X_scaled)
            
            # Identify Bull State
            # Calculate mean return for each state from ORIGINAL log_ret
            state_means = []
            for i in range(3):
                mask = (hidden_states == i)
                if np.sum(mask) > 0:
                    avg_ret = valid_data.loc[valid_data.index[mask], 'log_ret'].mean()
                    state_means.append(avg_ret)
                else:
                    state_means.append(-999)
            
            bull_state = np.argmax(state_means)
            
            regime = np.where(hidden_states == bull_state, 1, 0)
            
            full_regime = pd.Series(0, index=df.index)
            full_regime.loc[valid_data.index] = regime
            
            return full_regime.values
            
        except Exception as e:
            print(f"  [HMM Enhanced Error] {e}")
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values

    @staticmethod
    def detect_random_forest(df, market_name='Default'):
        """
        [NEW] Random Forest Regime Detector (Supervised)
        Tries to load pre-trained model from .pkl first.
        Falls back to on-the-fly training if model not found.
        """
        import os
        import joblib
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_predict
            try:
                from model.features import calculate_features
            except ImportError:
                from features import calculate_features  # When importing from model/
        except ImportError:
            print("  [Warning] sklearn not found. Falling back to SMA200.")
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values

        # Paths for pre-trained models
        model_dir = os.path.join(os.path.dirname(__file__), '../all_trad/regime_models')
        model_path = os.path.join(model_dir, f'regime_rf_{market_name}.pkl')
        features_path = os.path.join(model_dir, f'regime_features_{market_name}.pkl')
        
        # 1. Calculate Features (always needed)
        df_feat = calculate_features(df.copy())
        
        # 2. Try to load pre-trained model
        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                rf = joblib.load(model_path)
                valid_features = joblib.load(features_path)
                # print(f"  [RF] Loaded pre-trained model for {market_name}")
                
                # Verify features exist in df
                valid_features = [f for f in valid_features if f in df_feat.columns]
                
                if valid_features:
                    # Prepare X for prediction
                    X_full = df_feat[valid_features].values
                    X_full = np.nan_to_num(X_full)
                    
                    # Predict
                    full_preds = rf.predict(X_full)
                    
                    final_regime = pd.Series(full_preds, index=df.index)
                    return final_regime.values
                    
            except Exception as e:
                print(f"  [RF Warning] Error loading model for {market_name}: {e}")
        
        # 3. Fallback: Train on-the-fly (if no pre-trained model)
        # print(f"  [RF] Training on-the-fly for {market_name}")
        
        # Load features from CSV
        csv_path = os.path.join(os.path.dirname(__file__), f'../all_trad/regime_features_{market_name}.csv')
        selected_features = []
        if os.path.exists(csv_path):
            try:
                feature_df = pd.read_csv(csv_path)
                selected_features = feature_df['Feature'].tolist()
            except:
                pass
        
        if not selected_features:
            selected_features = [
                'volume_nvi', 'volume_obv', 'vma_120', 'volume_vpt', 'ma_120', 'vma_60',
                'p_ma_dist_60_120', 'trend_visual_ichimoku_b', 'vma_20', 'volume_adi', 'ma_60'
            ]
        
        valid_features = [f for f in selected_features if f in df_feat.columns]
        
        if not valid_features:
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values

        # Create Target
        future_ret = df_feat['Close'].shift(-20) / df_feat['Close'] - 1
        ensure_target = (future_ret > 0).astype(int)
        
        valid_mask = ~future_ret.isna()
        X = df_feat.loc[valid_mask, valid_features].values
        y = ensure_target.loc[valid_mask].values
        X = np.nan_to_num(X)
        
        if len(X) < 200:
             return np.ones(len(df))

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        X_full = df_feat[valid_features].values
        X_full = np.nan_to_num(X_full)
        full_preds = rf.predict(X_full)
        
        final_regime = pd.Series(full_preds, index=df.index)
        return final_regime.values

    @staticmethod
    def _detect_sklearn_model(df, market_name, model_prefix):
        import os
        import joblib
        from model.features import calculate_features
        
        model_dir = os.path.join(os.path.dirname(__file__), '../all_trad/regime_models')
        model_path = os.path.join(model_dir, f'regime_{model_prefix}_{market_name}.pkl')
        features_path = os.path.join(model_dir, f'regime_features_{market_name}.pkl')
        scaler_path = os.path.join(model_dir, f'regime_{model_prefix}_scaler_{market_name}.pkl')
        
        if not (os.path.exists(model_path) and os.path.exists(features_path)):
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values
            
        try:
            model = joblib.load(model_path)
            valid_features = joblib.load(features_path)
            
            df_feat = calculate_features(df.copy())
            valid_features = [f for f in valid_features if f in df_feat.columns]
            
            if not valid_features:
                 return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values
                 
            X_full = df_feat[valid_features].values
            X_full = np.nan_to_num(X_full)
            
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                X_full = scaler.transform(X_full)
                
            full_preds = model.predict(X_full)
            return pd.Series(full_preds, index=df.index).values
        except Exception as e:
            print(f"  [{model_prefix.upper()} Warning] Error loading model for {market_name}: {e}")
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values

    @staticmethod
    def detect_xgboost(df, market_name='Default'):
        return RegimeDetector._detect_sklearn_model(df, market_name, 'xgb')

    @staticmethod
    def detect_svc(df, market_name='Default'):
        return RegimeDetector._detect_sklearn_model(df, market_name, 'svc')

    @staticmethod
    def detect_logistic_regression(df, market_name='Default'):
        return RegimeDetector._detect_sklearn_model(df, market_name, 'lr')
        
    @staticmethod
    def detect_best_regime(df, market_name='Default'):
        """
        Loads the best regime model based on Pareto+ASF evaluation from MOO script.
        Defaults to SMA200 if not found.
        """
        import os
        import joblib
        
        model_dir = os.path.join(os.path.dirname(__file__), '../all_trad/regime_models')
        meta_path = os.path.join(model_dir, f'best_regime_meta_{market_name}.pkl')
        
        best_model = 'SMA200'
        if os.path.exists(meta_path):
            try:
                meta = joblib.load(meta_path)
                best_model = meta.get('best_model_type', 'SMA200')
            except:
                pass
                
        if best_model == 'SMA200':
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values
        elif best_model == 'ADX_Supertrend':
            return RegimeDetector.detect_adx_supertrend(df)
        elif best_model == 'GMM':
            return RegimeDetector.detect_gmm(df, market_name=market_name)
        elif best_model == 'HMM':
            return RegimeDetector.detect_hmm(df, market_name=market_name)
        elif best_model == 'RandomForest':
            return RegimeDetector.detect_random_forest(df, market_name=market_name)
        elif best_model == 'XGBoost':
            return RegimeDetector.detect_xgboost(df, market_name=market_name)
        elif best_model == 'SVC':
            return RegimeDetector.detect_svc(df, market_name=market_name)
        elif best_model == 'LogisticRegression':
            return RegimeDetector.detect_logistic_regression(df, market_name=market_name)
        else:
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values

    # ========================================================================
    # ENHANCED REGIME DETECTION  (Ensemble + Smoothing + Persistence + Probability)
    # ========================================================================

    @staticmethod
    def smooth_regime(regime, window=5, threshold=0.5):
        """Majority-vote smoothing over rolling window. Reduces whipsaws.

        regime    : array-like of 0/1
        window    : rolling lookback (centered)
        threshold : fraction of 1s required to mark as uptrend
        """
        s = pd.Series(np.asarray(regime, dtype=float))
        smooth = s.rolling(window=window, min_periods=1).mean()
        return (smooth >= threshold).astype(int).values

    @staticmethod
    def apply_persistence(regime, min_persist=3, persist_to_uptrend=None, persist_to_downtrend=None):
        """Require N consecutive flipped values before changing the active regime.

        Asymmetric variant (recommended for trading):
          - persist_to_downtrend = 1  (react fast to protect capital from drawdowns)
          - persist_to_uptrend   = 3  (require confirmation before re-entering long)

        If specific values are provided, they override min_persist.
        """
        regime = np.asarray(regime, dtype=int)
        if len(regime) == 0:
            return regime

        # Default to symmetric if not specified
        p_up = persist_to_uptrend if persist_to_uptrend is not None else min_persist
        p_down = persist_to_downtrend if persist_to_downtrend is not None else min_persist

        out = regime.copy()
        current = regime[0]
        flip_count = 0
        flip_target = current
        for i in range(len(regime)):
            if regime[i] != current:
                # We saw a flipped value
                if flip_target == regime[i]:
                    flip_count += 1
                else:
                    flip_target = regime[i]
                    flip_count = 1
                # Required consecutive flips depends on direction
                required = p_up if flip_target == 1 else p_down
                if flip_count >= required:
                    current = flip_target
                    flip_count = 0
            else:
                flip_count = 0
            out[i] = current
        return out

    @staticmethod
    def detect_ensemble(df, market_name='Default', methods=None, weights=None,
                        threshold=0.5, smooth_window=5, persistence=3):
        """Weighted-vote ensemble of multiple regime detectors.

        methods : list of method names. Defaults to a robust 5-detector mix.
        weights : optional dict {method_name: weight}. Defaults from regime_evaluation_summary.csv.

        Returns: regime array (0/1) AFTER smoothing + persistence filter.
        """
        import os
        import joblib

        if methods is None:
            methods = ['HMM', 'GMM', 'SMA200', 'RandomForest', 'XGBoost']

        # Load per-method weights from evaluation CSV (Sharpe-based)
        if weights is None:
            weights = RegimeDetector._load_eval_weights(market_name, methods)

        # Compute regimes from each method
        method_results = {}
        for m in methods:
            try:
                if m == 'HMM':
                    r = RegimeDetector.detect_hmm(df, market_name=market_name)
                elif m == 'GMM':
                    r = RegimeDetector.detect_gmm(df, market_name=market_name)
                elif m == 'SMA200':
                    sma = df['Close'].rolling(200).mean()
                    r = (df['Close'] > sma).astype(int).fillna(0).values
                elif m == 'ADX_Supertrend':
                    r = RegimeDetector.detect_adx_supertrend(df)
                elif m == 'RandomForest':
                    r = RegimeDetector.detect_random_forest(df, market_name=market_name)
                elif m == 'XGBoost':
                    r = RegimeDetector.detect_xgboost(df, market_name=market_name)
                elif m == 'SVC':
                    r = RegimeDetector.detect_svc(df, market_name=market_name)
                elif m == 'LogisticRegression':
                    r = RegimeDetector.detect_logistic_regression(df, market_name=market_name)
                else:
                    continue
                method_results[m] = np.asarray(r, dtype=float)
            except Exception as e:
                print(f"  [Ensemble Warning] {m} failed: {e}")
                continue

        if not method_results:
            sma = df['Close'].rolling(200).mean()
            return (df['Close'] > sma).astype(int).fillna(0).values

        # Weighted vote (continuous probability)
        n = len(df)
        weighted_sum = np.zeros(n)
        weight_total = 0.0
        for m, r in method_results.items():
            w = max(0.0, weights.get(m, 1.0))
            if len(r) != n:
                # Pad/truncate defensively
                if len(r) < n:
                    pad = np.zeros(n - len(r))
                    r = np.concatenate([pad, r])
                else:
                    r = r[-n:]
            weighted_sum += w * r
            weight_total += w

        if weight_total <= 0:
            weight_total = 1.0
        prob_uptrend = weighted_sum / weight_total

        # Threshold to binary, then smooth + persistence
        binary = (prob_uptrend >= threshold).astype(int)
        if smooth_window and smooth_window > 1:
            binary = RegimeDetector.smooth_regime(binary, window=smooth_window, threshold=0.5)
        if persistence and persistence > 1:
            binary = RegimeDetector.apply_persistence(binary, min_persist=persistence)

        return binary

    @staticmethod
    def detect_ensemble_proba(df, market_name='Default', methods=None, weights=None):
        """Same as detect_ensemble but returns CONTINUOUS probability [0,1] without thresholding.

        Useful for position-sizing strategies (e.g., scale long position by regime confidence).
        """
        import os
        import joblib

        if methods is None:
            methods = ['HMM', 'GMM', 'SMA200', 'RandomForest', 'XGBoost']
        if weights is None:
            weights = RegimeDetector._load_eval_weights(market_name, methods)

        method_results = {}
        for m in methods:
            try:
                if m == 'HMM':
                    r = RegimeDetector.detect_hmm(df, market_name=market_name)
                elif m == 'GMM':
                    r = RegimeDetector.detect_gmm(df, market_name=market_name)
                elif m == 'SMA200':
                    sma = df['Close'].rolling(200).mean()
                    r = (df['Close'] > sma).astype(int).fillna(0).values
                elif m == 'RandomForest':
                    r = RegimeDetector.detect_random_forest(df, market_name=market_name)
                elif m == 'XGBoost':
                    r = RegimeDetector.detect_xgboost(df, market_name=market_name)
                else:
                    continue
                method_results[m] = np.asarray(r, dtype=float)
            except Exception:
                continue

        if not method_results:
            return np.full(len(df), 0.5)

        n = len(df)
        weighted_sum = np.zeros(n)
        weight_total = 0.0
        for m, r in method_results.items():
            w = max(0.0, weights.get(m, 1.0))
            if len(r) != n:
                if len(r) < n:
                    r = np.concatenate([np.zeros(n - len(r)), r])
                else:
                    r = r[-n:]
            weighted_sum += w * r
            weight_total += w

        if weight_total <= 0:
            weight_total = 1.0
        prob = weighted_sum / weight_total
        # EMA smoothing for stable probability
        prob_smooth = pd.Series(prob).ewm(span=5, adjust=False).mean().values
        return prob_smooth

    @staticmethod
    def _load_eval_weights(market_name, methods, min_sharpe=0.3, dominant_ratio=1.5):
        """Load per-method weights from regime_evaluation_summary.csv.

        Strategy:
          1. Best method (Is_Best=True in CSV, OR highest composite quality) gets weight 1.0
          2. Other methods with Sharpe >= min_sharpe get weight 0.25 * Sharpe^2 (small support)
          3. Methods with Sharpe < min_sharpe get weight 0
          4. If best is dominantly better (ratio >= dominant_ratio), winner-take-all

        This ensures the per-market best method always leads, with light ensemble support
        only from genuinely useful methods.
        """
        import os

        csv_path = os.path.join(os.path.dirname(__file__),
                                '../all_trad/regime_models/regime_evaluation_summary.csv')
        weights = {m: 0.0 for m in methods}

        if not os.path.exists(csv_path):
            return {m: 1.0 for m in methods}  # uniform fallback

        try:
            df_eval = pd.read_csv(csv_path)
            df_m = df_eval[df_eval['Market'] == market_name]
            if df_m.empty:
                return {m: 1.0 for m in methods}

            # Collect quality per method
            scored = []
            best_method_csv = None
            for m in methods:
                row = df_m[df_m['Method'] == m]
                if row.empty:
                    continue
                sharpe = float(row['Sharpe'].iloc[0])
                sep = float(row['Separation'].iloc[0]) if 'Separation' in row.columns else 0.0
                is_best = bool(row['Is_Best'].iloc[0]) if 'Is_Best' in row.columns else False
                if is_best:
                    best_method_csv = m
                quality = sharpe + 2.0 * sep
                scored.append((m, sharpe, sep, quality))

            if not scored:
                return {m: 1.0 for m in methods}

            scored.sort(key=lambda x: -x[3])
            best_method = best_method_csv if best_method_csv else scored[0][0]
            best_quality = scored[0][3]
            second_quality = scored[1][3] if len(scored) > 1 else 0.0

            # Winner-take-all if dominant
            if best_quality > 0 and (
                second_quality <= 0 or
                best_quality / max(abs(second_quality), 0.1) >= dominant_ratio
            ):
                for m in methods:
                    weights[m] = 0.0
                weights[best_method] = 1.0
                return weights

            # Otherwise: best method 1.0, supporting methods 0.25*Sharpe^2
            for m, sharpe, sep, quality in scored:
                if m == best_method:
                    weights[m] = 1.0
                elif sharpe >= min_sharpe:
                    weights[m] = 0.25 * (sharpe ** 2)
                else:
                    weights[m] = 0.0
        except Exception as e:
            print(f"  [Ensemble Warning] eval-weight load failed: {e}")
            return {m: 1.0 for m in methods}
        return weights

    @staticmethod
    def detect_super_regime(df, market_name='Default', smooth_window=3,
                            persist_to_uptrend=3, persist_to_downtrend=1,
                            noise_threshold=20):
        """Best-quality regime detection — adaptive between best-single and ensemble.

        Logic:
          1. Compute the per-market BEST regime (from CSV, e.g. HMM for UK/US).
          2. If the best regime has FEW flips (clean signal) → return it raw.
          3. If the best regime is NOISY (>= noise_threshold flips) → apply ensemble +
             smoothing + asymmetric persistence to clean it up.

        Asymmetric persistence (when applied):
          - 1 day to flip → downtrend (react fast to protect capital)
          - 3 days to flip → uptrend (confirm rally)

        Args:
          noise_threshold: max flips in detected period. Below this we trust the model;
                           above this we smooth + apply persistence.
        """
        # Step 1: Get the per-market best regime
        best_regime = RegimeDetector.detect_best_regime(df, market_name=market_name)
        best_arr = np.asarray(best_regime, dtype=int)
        flips = int(np.sum(np.abs(np.diff(best_arr))))
        n = len(best_arr)
        up_ratio = float(np.mean(best_arr)) if n > 0 else 0.5

        # Step 2: If best signal is "useless" (all-same or near-all-same), use ensemble
        is_degenerate = (up_ratio < 0.05) or (up_ratio > 0.95) or (flips == 0)

        # Step 3: If clean and not degenerate, trust the best single method
        if flips < noise_threshold and not is_degenerate:
            return best_arr

        # Step 3: Otherwise, build ensemble & clean up
        prob = RegimeDetector.detect_ensemble_proba(
            df, market_name=market_name,
            methods=['HMM', 'GMM', 'SMA200', 'RandomForest', 'XGBoost'],
        )
        binary = (prob >= 0.5).astype(int)
        if smooth_window and smooth_window > 1:
            binary = RegimeDetector.smooth_regime(binary, window=smooth_window, threshold=0.5)
        binary = RegimeDetector.apply_persistence(
            binary,
            persist_to_uptrend=persist_to_uptrend,
            persist_to_downtrend=persist_to_downtrend,
        )
        return binary
