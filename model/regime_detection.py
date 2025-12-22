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
    def detect_gmm(df, window=20):
        """
        Uses GMM to cluster market into 2 states based on Log Returns and Volatility.
        State with higher mean return (or lower volatility if means are close) is Uptrend (1).
        """
        data = df.copy()
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['log_ret'].rolling(window=window).std()
        
        # Drop NaNs
        valid_data = data.dropna().copy()
        
        if len(valid_data) < 100:
            return np.ones(len(df)) # Default to Uptrend if not enough data
            
        X = valid_data[['log_ret', 'volatility']].values
        
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)
            labels = gmm.predict(X)
        except Exception as e:
            print(f"  [GMM Warning] Fit failed ({e}). Falling back to SMA200.")
            sma = data['Close'].rolling(200).mean()
            return (data['Close'] > sma).astype(int).fillna(0).values
        
        # Identify which label is 'Uptrend'
        # Uptrend usually has positive mean return (or lower vol in some contexts, but let's stick to return)
        # Check mean return of cluster 0 vs 1
        mean_ret_0 = X[labels == 0, 0].mean()
        mean_ret_1 = X[labels == 1, 0].mean()
        
        uptrend_label = 0 if mean_ret_0 > mean_ret_1 else 1
        
        # Map labels to 0/1 (1 = Uptrend)
        regime = np.where(labels == uptrend_label, 1, 0)
        
        # Realign with original index
        full_regime = pd.Series(0, index=df.index) # Default Downtrend for NaNs
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
    def detect_hmm(df):
        """
        Uses Hidden Markov Model to infer hidden states (Bull/Bear) from Returns.
        Requires 'hmmlearn'. Returns SMA200 fallback if hmmlearn missing.
        """
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
        
        # Fit HMM
        # n_components=2 (Bull/Bear)
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X)
        hidden_states = model.predict(X)
        
        # Identify Bull state
        # State with higher mean return
        means = model.means_[:, 0]
        bull_state = np.argmax(means)
        
        regime = np.where(hidden_states == bull_state, 1, 0)
        
        full_regime = pd.Series(0, index=df.index)
        full_regime.loc[valid_data.index] = regime
        
        return full_regime.values
