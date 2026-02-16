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
            print("  [Warning] hmmlearn not found. Falling back to SMA200.")
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values

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
            print("  [Warning] hmmlearn not found. Falling back to SMA200.")
            return (df['Close'] > df['Close'].rolling(200).mean()).astype(int).fillna(0).values
            
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
            from model.features import calculate_features
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
