"""
Trading System — ระบบเทรดอัตโนมัติ
===================================
สกัดจาก backtest_all_models.py

ประกอบด้วย 3 ส่วนหลัก:
1. TrendDetector  — ตรวจจับ Uptrend/Downtrend (HMM, GMM, ADX+Supertrend)
2. SignalGenerator — สร้างสัญญาณ BUY / SELL / HOLD
3. TradingEngine   — จำลองเทรดพร้อม Stop Loss, Trailing Stop

Usage:
    python workflows/trading_system.py                  # แสดง Trend + สัญญาณปัจจุบัน
    python workflows/trading_system.py --market US      # ดูเฉพาะตลาด US
    python workflows/trading_system.py --backtest       # รัน backtest
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
import yfinance as yf
import sqlite3
import datetime

warnings.filterwarnings('ignore')

# Add project root to sys.path to allow importing from model/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.join(current_script_dir, '..')
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from model.backtest_all_models import run_simulation_moo

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

MARKETS = ['US', 'UK', 'Thai', 'Gold', 'BTC']

TICKERS = {
    'US': '^GSPC',
    'UK': '^FTSE',
    'Thai': '^SET.BK',
    'Gold': 'GC=F',
    'BTC': 'BTC-USD',
}

# ─────────────────────────────────────────────────────────────────
# DYNAMIC CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────
# Global variable states holding the best-performing models & parameters
MARKET_REGIME_METHOD = {}
MARKET_UPTREND_MODEL = {}
MARKET_DOWNTREND_MODEL = {}
OPTIMIZED_PARAMS = {}
MARKET_STRATEGIES = {}
MARKET_LONG_ONLY = {}
MARKET_LEVERAGE = {}
MARKET_SIZING_MODE = {}
MARKET_MIN_POSITION = {}

# Default Trailing Stop / Long-Only overrides (unless overridden by ML later)
MARKET_TRAILING_STOPS = {
    'US':   0.15, 'UK':   0.10, 'Gold': 0.15, 'BTC':  0.20, 'Thai': 0.10,
}

# MARKET_LONG_ONLY is loaded dynamically from backtest_combined_results.csv via load_dynamic_configs()

LOOKBACK = 30
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001

def load_dynamic_configs(quiet=True):
    """
    Load parameters and model selections dynamically from 
    'backtest_combined_results.csv' to ensure we utilize the absolute
    best-performing configurations from our most recent MOO optimizations.
    """
    global MARKET_REGIME_METHOD, MARKET_UPTREND_MODEL, MARKET_DOWNTREND_MODEL
    global OPTIMIZED_PARAMS, MARKET_STRATEGIES
    global MARKET_LONG_ONLY, MARKET_LEVERAGE, MARKET_TRAILING_STOPS
    global MARKET_SIZING_MODE, MARKET_MIN_POSITION

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    combined_csv = os.path.join(project_root, 'backtest_combined_results.csv')
    
    if not os.path.exists(combined_csv):
        if not quiet:
            print(f"  [Warning] Dynamic config {combined_csv} not found.")
        return

    try:
        df = pd.read_csv(combined_csv)
        # Filter strictly for rows denoting the hybrid MOO logic
        moo_df = df[df['Type'].str.contains('MOO', na=False)]
        
        for idx, row in moo_df.iterrows():
            market = row['Market']
            
            # Extract models
            MARKET_REGIME_METHOD[market] = row['Regime Method']
            MARKET_UPTREND_MODEL[market] = row['Uptrend Model']
            MARKET_DOWNTREND_MODEL[market] = row['Downtrend Model']
            
            # Extract strategy parsing from string e.g., 'MOO (active)[FB:True]' -> 'active'
            type_str = row['Type']
            strategy_mode = 'active'
            if '(smart_hold)' in type_str:
                strategy_mode = 'smart_hold'
            elif '(Buy_and_Hold_Override)' in type_str:
                strategy_mode = 'Buy_and_Hold_Override'
            elif '(Enhanced_Oracle_B&H)' in type_str:
                strategy_mode = 'Enhanced_Oracle_B&H'
            elif '(Perfect_Oracle)' in type_str:
                strategy_mode = 'Perfect_Oracle'
            
            fallback_mode = False
            if '[FB:True]' in type_str:
                fallback_mode = True
            
            MARKET_STRATEGIES[market] = strategy_mode
            
            # Extract parameters
            OPTIMIZED_PARAMS[market] = {
                'confidence': float(row['Confidence']),
                'long': float(row['Threshold Long']),
                'short': float(row['Threshold Short']),
                'stop_loss': float(row['Stop Loss']),
                'fallback': fallback_mode
            }
            
            # Extract Long Only if present in Type string or fallback to False
            if '[LO:True]' in type_str:
                MARKET_LONG_ONLY[market] = True
            elif '[LO:False]' in type_str:
                MARKET_LONG_ONLY[market] = False
                
            # Extract Leverage if present in Type string (e.g., [Lev:1.5])
            import re
            match = re.search(r'\[Lev:([0-9.]+)\]', type_str)
            if match:
                MARKET_LEVERAGE[market] = float(match.group(1))
            else:
                MARKET_LEVERAGE[market] = 1.0
            
            # Extract Trailing Stop from CSV column if it exists
            if 'Trailing' in row.index:
                MARKET_TRAILING_STOPS[market] = float(row['Trailing'])
            
            # Extract Sizing Mode (e.g., [SM:aggressive])
            sm_match = re.search(r'\[SM:(\w+)\]', type_str)
            if sm_match:
                MARKET_SIZING_MODE[market] = sm_match.group(1)
            else:
                MARKET_SIZING_MODE[market] = 'conservative'
            
            # Extract Min Position (e.g., [MP:0.3])
            mp_match = re.search(r'\[MP:([0-9.]+)\]', type_str)
            if mp_match:
                MARKET_MIN_POSITION[market] = float(mp_match.group(1))
            else:
                MARKET_MIN_POSITION[market] = 0.0
            
        if not quiet:
            print(f"  ✔️ Successfully loaded dynamic configurations for {len(MARKET_STRATEGIES)} markets.")
            
    except Exception as e:
        if not quiet:
            print(f"  [Error] Failed to load dynamic configs: {e}")

# Pre-warm configurations globally upon module load
load_dynamic_configs()


# ═════════════════════════════════════════════════════════════════
# 1. TREND DETECTOR — ตรวจจับเทรนด์ขึ้น/ลง
# ═════════════════════════════════════════════════════════════════

class TrendDetector:
    """
    ตรวจจับ Market Regime (Uptrend=1 / Downtrend=0)
    
    3 วิธี:
    - HMM:              Hidden Markov Model บน log returns
    - GMM:              Gaussian Mixture Model บน returns + volatility 
    - ADX + Supertrend: ADX > 25 + Supertrend สีเขียว
    
    ผลลัพธ์: numpy array ของ 0 (Downtrend) หรือ 1 (Uptrend) ขนาดเดียวกับ DataFrame
    """

    @staticmethod
    def detect(df, method='HMM', market_name='Default'):
        """
        ตรวจจับ trend ด้วยวิธีที่เลือก
        
        Parameters:
            df: DataFrame ที่มี OHLCV
            method: 'HMM', 'GMM', 'HMM_Enhanced', 'GMM_Enhanced', 'ADX_Supertrend'
            market_name: ชื่อตลาด (ใช้โหลด pre-trained model)
            
        Returns:
            numpy array: 1=Uptrend, 0=Downtrend
        """
        method = method.upper().replace(' ', '_')
        
        if method == 'HMM':
            return TrendDetector.detect_hmm(df, market_name)
        elif method == 'GMM':
            return TrendDetector.detect_gmm(df, market_name)
        elif method == 'HMM_ENHANCED':
            return TrendDetector.detect_hmm_enhanced(df)
        elif method == 'GMM_ENHANCED':
            return TrendDetector.detect_gmm_enhanced(df)
        elif method in ('ADX_SUPERTREND', 'ADX'):
            return TrendDetector.detect_adx_supertrend(df)
        elif method == 'SMA200':
            return TrendDetector.detect_sma200(df)
        else:
            import sys as _sys
            print(f"  [Warning] Unknown method '{method}'. Using HMM.", file=_sys.stderr)
            return TrendDetector.detect_hmm(df, market_name)

    # ─── HMM ────────────────────────────────────────────────────
    @staticmethod
    def detect_hmm(df, market_name='Default'):
        """
        Hidden Markov Model บน Log Returns
        - พยายามโหลด pre-trained model ก่อน
        - ถ้าไม่มี จะ train on-the-fly
        """
        import joblib
        
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
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

        # Try pre-trained model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, '../all_trad/regime_models')
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
                print(f"  [HMM] Pre-trained load error: {e}")

        # Fallback: train on-the-fly
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        try:
            model.fit(X)
        except Exception:
            return np.ones(len(df))

        hidden_states = model.predict(X)
        means = model.means_[:, 0]
        bull_state = np.argmax(means)
        regime = np.where(hidden_states == bull_state, 1, 0)

        full_regime = pd.Series(0, index=df.index)
        full_regime.loc[valid_data.index] = regime
        return full_regime.values

    # ─── GMM ────────────────────────────────────────────────────
    @staticmethod
    def detect_gmm(df, market_name='Default', window=20):
        """
        Gaussian Mixture Model บน Log Returns + Volatility
        """
        import joblib
        from sklearn.mixture import GaussianMixture

        data = df.copy()
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['log_ret'].rolling(window=window).std()
        valid_data = data.dropna().copy()

        if len(valid_data) < 100:
            return np.ones(len(df))

        X = valid_data[['log_ret', 'volatility']].values

        # Try pre-trained model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, '../all_trad/regime_models')
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
                print(f"  [GMM] Pre-trained load error: {e}")

        # Fallback: train on-the-fly
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)
            labels = gmm.predict(X)
        except Exception:
            return np.ones(len(df))

        mean_ret_0 = X[labels == 0, 0].mean()
        mean_ret_1 = X[labels == 1, 0].mean()
        uptrend_label = 0 if mean_ret_0 > mean_ret_1 else 1

        regime = np.where(labels == uptrend_label, 1, 0)
        full_regime = pd.Series(0, index=df.index)
        full_regime.loc[valid_data.index] = regime
        return full_regime.values

    # ─── GMM Enhanced (3 States) ────────────────────────────────
    @staticmethod
    def detect_gmm_enhanced(df, window=20):
        """GMM แบบ 3 States + RSI + ADX features"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.mixture import GaussianMixture
        from ta.momentum import RSIIndicator
        from ta.trend import ADXIndicator

        data = df.copy()
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['log_ret'].rolling(window=window).std()
        data['rsi'] = RSIIndicator(data['Close'], window=14).rsi()
        data['adx'] = ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()

        valid_data = data.dropna().copy()
        if len(valid_data) < 200:
            return np.ones(len(df))

        X = valid_data[['log_ret', 'volatility', 'rsi', 'adx']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
            gmm.fit(X_scaled)
            labels = gmm.predict(X_scaled)

            cluster_means = []
            for i in range(3):
                mask = (labels == i)
                if np.sum(mask) > 0:
                    cluster_means.append(valid_data.loc[valid_data.index[mask], 'log_ret'].mean())
                else:
                    cluster_means.append(-999)

            uptrend_state = np.argmax(cluster_means)
            regime = np.where(labels == uptrend_state, 1, 0)

            full_regime = pd.Series(0, index=df.index)
            full_regime.loc[valid_data.index] = regime
            return full_regime.values

        except Exception as e:
            print(f"  [GMM Enhanced Error] {e}")
            return np.ones(len(df))

    # ─── HMM Enhanced (3 States) ────────────────────────────────
    @staticmethod
    def detect_hmm_enhanced(df):
        """HMM แบบ 3 States + Volatility feature"""
        from sklearn.preprocessing import StandardScaler

        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            import subprocess
            print("  [Warning] hmmlearn not found. Auto-installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn"])
            from hmmlearn.hmm import GaussianHMM

        data = df.copy()
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['log_ret'].rolling(window=20).std()
        valid_data = data.dropna().copy()

        if len(valid_data) < 200:
            return np.ones(len(df))

        X = valid_data[['log_ret', 'volatility']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            model.fit(X_scaled)
            hidden_states = model.predict(X_scaled)

            state_means = []
            for i in range(3):
                mask = (hidden_states == i)
                if np.sum(mask) > 0:
                    state_means.append(valid_data.loc[valid_data.index[mask], 'log_ret'].mean())
                else:
                    state_means.append(-999)

            bull_state = np.argmax(state_means)
            regime = np.where(hidden_states == bull_state, 1, 0)

            full_regime = pd.Series(0, index=df.index)
            full_regime.loc[valid_data.index] = regime
            return full_regime.values

        except Exception as e:
            print(f"  [HMM Enhanced Error] {e}")
            return np.ones(len(df))

    # ─── ADX + Supertrend ───────────────────────────────────────
    @staticmethod
    def detect_adx_supertrend(df, adx_period=14, adx_threshold=25, atr_period=10, multiplier=3.0):
        """
        Uptrend เมื่อ ADX > 25 AND Supertrend เป็นสีเขียว (Close > Supertrend)
        """
        from ta.trend import ADXIndicator
        from ta.volatility import AverageTrueRange

        high = df['High']
        low = df['Low']
        close = df['Close']

        adx = ADXIndicator(high, low, close, window=adx_period).adx()
        atr = AverageTrueRange(high, low, close, window=atr_period).average_true_range()

        hl2 = (high + low) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        final_upper = pd.Series(0.0, index=df.index)
        final_lower = pd.Series(0.0, index=df.index)
        supertrend = pd.Series(0.0, index=df.index)

        final_upper.iloc[0] = basic_upper.iloc[0]
        final_lower.iloc[0] = basic_lower.iloc[0]

        for i in range(1, len(df)):
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]

            if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]

            prev_st = supertrend.iloc[i-1]
            if prev_st == 1:
                supertrend.iloc[i] = -1 if close.iloc[i] <= final_lower.iloc[i] else 1
            elif prev_st == -1:
                supertrend.iloc[i] = 1 if close.iloc[i] >= final_upper.iloc[i] else -1
            else:
                supertrend.iloc[i] = 1 if close.iloc[i] > close.iloc[i-1] else -1

        is_uptrend = ((adx > adx_threshold) & (supertrend == 1)).astype(int)
        return is_uptrend.values

    # ─── SMA200 ─────────────────────────────────────────────────
    @staticmethod
    def detect_sma200(df):
        """
        Simple Moving Average 200-day Regime Detection.
        Uptrend when Close > SMA(200), Downtrend otherwise.
        """
        sma200 = df['Close'].rolling(200).mean()
        regime = (df['Close'] > sma200).astype(int).fillna(0)
        return regime.values

    # ─── Utility ────────────────────────────────────────────────
    @staticmethod
    def get_current_trend(df, method='HMM', market_name='Default'):
        """
        คืนค่า trend ล่าสุด: 'UPTREND' หรือ 'DOWNTREND'
        """
        regime = TrendDetector.detect(df, method, market_name)
        current = regime[-1] if len(regime) > 0 else 0
        return 'UPTREND' if current == 1 else 'DOWNTREND'

    @staticmethod
    def get_trend_summary(df, method='HMM', market_name='Default'):
        """
        คืน summary ของ trend: จำนวนวัน uptrend/downtrend, % uptrend
        """
        regime = TrendDetector.detect(df, method, market_name)
        up_days = int(np.sum(regime == 1))
        down_days = int(np.sum(regime == 0))
        total = len(regime)
        pct_up = (up_days / total * 100) if total > 0 else 0
        current = 'UPTREND' if regime[-1] == 1 else 'DOWNTREND'

        return {
            'current_trend': current,
            'uptrend_days': up_days,
            'downtrend_days': down_days,
            'uptrend_pct': round(pct_up, 1),
            'total_days': total,
        }


# ═════════════════════════════════════════════════════════════════
# 2. SIGNAL GENERATOR — สร้างสัญญาณซื้อขาย
# ═════════════════════════════════════════════════════════════════

class SignalGenerator:
    """
    สร้างสัญญาณซื้อขายจาก model predictions
    
    สัญญาณ:
        1  = BUY (Long)     — มั่นใจว่าขึ้น
        0  = SELL (Short)   — มั่นใจว่าลง
        99 = HOLD (Neutral) — ไม่มั่นใจ → ไม่ทำอะไร
    
    Parameters (ต่อตลาด):
        confidence: ขั้นต่ำของความเชื่อมั่นในการเปิดสถานะ
        long:       threshold ของ bull_prob สำหรับ BUY
        short:      threshold ของ bull_prob สำหรับ SELL
    """

    def __init__(self, market='US'):
        params = OPTIMIZED_PARAMS.get(market, {
            'confidence': 0.55, 'long': 0.54, 'short': 0.46, 'fallback': False
        })
        self.confidence_threshold = params['confidence']
        self.long_threshold = params['long']
        self.short_threshold = params['short']
        self.fallback = params.get('fallback', False)
        self.market = market

    def generate(self, probabilities):
        """
        สร้างสัญญาณจาก probability array
        
        Parameters:
            probabilities: numpy array shape (N, n_classes)
                          โดย column 1 = bull probability
                          
        Returns:
            numpy array ของสัญญาณ (1=BUY, 0=SELL, 99=HOLD)
        """
        signals = []

        for i in range(len(probabilities)):
            prob = probabilities[i]
            confidence = np.max(prob)
            bull_prob = prob[1] if len(prob) > 1 else prob[0]

            signal = self._classify(confidence, bull_prob)
            signals.append(signal)

        return np.array(signals)

    def generate_from_regime(self, probs_uptrend, probs_downtrend, regime):
        """
        สร้างสัญญาณโดยเลือก model ตาม regime
        
        Parameters:
            probs_uptrend:   predictions จาก uptrend model
            probs_downtrend: predictions จาก downtrend model
            regime:          array ของ 0/1 (0=downtrend, 1=uptrend)
            
        Returns:
            numpy array ของสัญญาณ
        """
        signals = []

        for i in range(len(regime)):
            if regime[i] == 1:
                prob = probs_uptrend[i]
            else:
                prob = probs_downtrend[i]

            confidence = np.max(prob)
            bull_prob = prob[1] if len(prob) > 1 else prob[0]
            signal = self._classify(confidence, bull_prob, regime[i])
            signals.append(signal)

        return np.array(signals)

    def _classify(self, confidence, bull_prob, regime_val=None):
        """ตัดสินสัญญาณจาก confidence และ bull_prob"""
        if confidence < self.confidence_threshold:
            return regime_val if (self.fallback and regime_val is not None) else 99  # HOLD — ไม่มั่นใจ
        elif bull_prob >= self.long_threshold:
            return 1   # BUY
        elif bull_prob <= self.short_threshold:
            return 0   # SELL
        else:
            return regime_val if (self.fallback and regime_val is not None) else 99  # HOLD — อยู่ในโซนกลาง

    @staticmethod
    def signal_to_text(signal):
        """แปลงตัวเลขสัญญาณเป็นข้อความ"""
        if signal == 1:
            return '🟢 BUY'
        elif signal == 0:
            return '🔴 SELL'
        else:
            return '⚪ HOLD'

    def get_params_text(self):
        """แสดง parameters ปัจจุบัน"""
        return (
            f"Confidence ≥ {self.confidence_threshold:.4f} | "
            f"BUY ≥ {self.long_threshold:.4f} | "
            f"SELL ≤ {self.short_threshold:.4f}"
        )


# ═════════════════════════════════════════════════════════════════
# 3. TRADING ENGINE — เครื่องยนต์เทรด
# ═════════════════════════════════════════════════════════════════

class TradingEngine:
    """
    จำลองการเทรดจากสัญญาณ
    
    Features:
        - Stop Loss:     ตัดขาดทุนอัตโนมัติ
        - Trailing Stop: ล็อคกำไรเมื่อราคาถอยจากจุดสูงสุด
        - 2 Modes:       'active' (neutral→cash) vs 'smart_hold' (neutral→hold)
        - Long Only:     option สำหรับตลาดที่ขึ้นเป็นหลัก
    """

    def __init__(self, market='US', initial_capital=INITIAL_CAPITAL):
        self.market = market
        self.initial_capital = initial_capital
        self.transaction_cost = TRANSACTION_COST

        params = OPTIMIZED_PARAMS.get(market, {'stop_loss': 0.05})
        self.stop_loss_pct = params['stop_loss']
        self.trailing_stop_pct = MARKET_TRAILING_STOPS.get(market, 0.0)
        self.long_only = MARKET_LONG_ONLY.get(market, False)
        self.strategy_mode = MARKET_STRATEGIES.get(market, 'active')
        self.leverage = MARKET_LEVERAGE.get(market, 1.0)
        self.sizing_mode = MARKET_SIZING_MODE.get(market, 'conservative')
        self.min_position = MARKET_MIN_POSITION.get(market, 0.0)

    def run(self, signals, prices, confidences=None, regimes=None):
        """
        รัน simulation
        
        Parameters:
            signals: numpy array (1=BUY, 0=SELL, 99=HOLD)
            prices:  numpy array ของราคาปิด
            
        Returns:
            dict: {
                'equity_curve': list,
                'total_return': float,
                'n_trades': int,
                'win_rate': float,
                'max_drawdown': float,
                'final_equity': float,
            }
        """
        # ─── B&H Override Checks Removed to ensure genuine signals ───

        equity_curve, n_trades, win_rate, trade_points = run_simulation_moo(
            signals=signals,
            prices=prices,
            confidences=confidences,
            regimes=regimes,
            stop_loss_pct=self.stop_loss_pct,
            trailing_stop_pct=self.trailing_stop_pct,
            long_only=self.long_only,
            strategy_mode=self.strategy_mode,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            leverage=self.leverage,
            sizing_mode=self.sizing_mode,
            min_position=self.min_position
        )

        # Calculate metrics
        cash = equity_curve[-1] if equity_curve else self.initial_capital
        total_return = (cash - self.initial_capital) / self.initial_capital * 100
        max_dd = self._calculate_drawdown(equity_curve)
        bnh_return = (prices[-1] - prices[0]) / prices[0] * 100

        return {
            'equity_curve': equity_curve,
            'total_return': round(total_return, 2),
            'bnh_return': round(bnh_return, 2),
            'n_trades': n_trades,
            'win_rate': round(win_rate, 1),
            'max_drawdown': round(max_dd, 2),
            'final_equity': round(cash, 2),
            'trade_points': trade_points,
        }

    @staticmethod
    def _calculate_drawdown(equity_curve):
        """คำนวณ Maximum Drawdown"""
        if not equity_curve:
            return 0.0
        eq = pd.Series(equity_curve)
        peak = eq.cummax()
        drawdown = (eq - peak) / peak
        return drawdown.min() * 100

    def get_config_text(self):
        """แสดง configuration ปัจจุบัน"""
        return (
            f"Strategy: {self.strategy_mode} | "
            f"SL: {self.stop_loss_pct:.1%} | "
            f"TS: {self.trailing_stop_pct:.1%} | "
            f"Long Only: {self.long_only}"
        )


# ═════════════════════════════════════════════════════════════════
# 4. CLI — แสดงสัญญาณปัจจุบัน
# ═════════════════════════════════════════════════════════════════

def download_market_data(market, period='2y'):
    """ดาวน์โหลดข้อมูลตลาด"""
    ticker = TICKERS.get(market)
    if not ticker:
        return None

    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except Exception as e:
        print(f"  Error downloading {market}: {e}")
        return None


def get_current_signals(markets=None, quiet=False):
    """
    ดึงสัญญาณปัจจุบันของแต่ละตลาดแบบ REAL ML PREDICTION
    
    Returns:
        list of dict: [{market, trend, signal, signal_text, price, date, ...}, ...]
    """
    import joblib
    import tensorflow as tf
    import os
    import sys
    import numpy as np
    
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model as keras_load_model
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    model_base_dir = os.path.join(project_root, 'all_trad', 'separate_models')
    sys.path.insert(0, os.path.join(project_root, 'model'))
    
    try:
        from backtest_all_models import calculate_features, get_selected_features
    except ImportError:
        if not quiet: print("❌ Cannot import from backtest_all_models.py")
        return []

    if markets is None:
        markets = MARKETS

    results = []

    for market in markets:
        regime_method = MARKET_REGIME_METHOD.get(market, 'HMM')
        up_model = MARKET_UPTREND_MODEL.get(market, '?')
        down_model = MARKET_DOWNTREND_MODEL.get(market, '?')
        strategy = MARKET_STRATEGIES.get(market, 'active')

        if not quiet:
            print(f"\n📊 {market} ({TICKERS[market]})")
            print(f"   ⚙️  Regime: {regime_method} | Models: ↑{up_model} ↓{down_model} | Strategy: {strategy}")
            print(f"   Downloading data & calculating features...")

        df = download_market_data(market, period='1y')
        if df is None:
            if not quiet:
                print(f"   ❌ No data available")
            results.append({
                'market': market, 'trend': 'N/A', 'signal': 'N/A', 'price': 'N/A', 'date': 'N/A'
            })
            continue

        try:
            df = calculate_features(df)
            df = df.dropna()
        except Exception as e:
            if not quiet: print(f"   ❌ Error calculating features: {e}")
            continue

        if df.empty:
            if not quiet: print(f"   ❌ Data is empty after TA calculation")
            continue

        # 1. Detect Trend (ใช้ regime method ที่ดีที่สุดสำหรับตลาดนี้)
        trend = TrendDetector.get_current_trend(df, method=regime_method, market_name=market)
        summary = TrendDetector.get_trend_summary(df, method=regime_method, market_name=market)

        # 2. Current Price
        current_price = df['Close'].iloc[-1]
        current_date = df.index[-1].strftime('%Y-%m-%d')
        
        # 3. Model Prediction
        active_model_name = up_model if trend == 'UPTREND' else down_model
        trend_suffix = 'uptrend' if trend == 'UPTREND' else 'downtrend'
        model_dir = os.path.join(model_base_dir, f"model_{market}_{trend_suffix}")
        
        if not quiet: print(f"   Loading active model: {active_model_name}...")
        
        try:
            model, scaler, le, model_type = _load_model_and_objects(model_dir, active_model_name, keras_load_model, joblib)
        except Exception as e:
            if not quiet: print(f"   ❌ Error loading model: {e}")
            model = None
            
        signal = 99
        signal_text = "⚪ HOLD (รอ)"
        ml_up_prob = 0.0
        ml_down_prob = 0.0
        
        if model is not None:
            selected_cols = get_selected_features(df)
            missing = [f for f in selected_cols if f not in df.columns]
            for m in missing: df[m] = 0
            feature_data = df[selected_cols].values
            
            check_lookback = LOOKBACK
            if model_type == 'ml' and hasattr(model, 'n_features_in_'):
                n_expected = model.n_features_in_
                n_feats = len(selected_cols)
                if n_feats > 0:
                    calculated_lb = n_expected // n_feats
                    if calculated_lb != LOOKBACK: check_lookback = calculated_lb
            
            if len(feature_data) > check_lookback:
                if scaler:
                    try: feature_data_scaled = scaler.transform(feature_data)
                    except: feature_data_scaled = feature_data
                else:
                    feature_data_scaled = feature_data
                    
                window = feature_data_scaled[-check_lookback:]
                X_input = window.flatten() if model_type == 'ml' else window
                X_input = np.array([X_input])
                
                # Predict probability
                if model_type == 'keras':
                    prob = model.predict(X_input, verbose=0)[0]
                else:
                    if hasattr(model, 'predict_proba'): prob = model.predict_proba(X_input)[0]
                    else: 
                        p = model.predict(X_input)[0]
                        prob = np.zeros(2)
                        prob[int(p)] = 1.0
                
                # Generate signal based on MOO thresholds
                sg = SignalGenerator(market)
                signals = sg.generate(np.array([prob]))
                signal = signals[0]
                
                # Store ML probabilities
                ml_up_prob = round(float(prob[1] * 100 if len(prob) > 1 else prob[0] * 100), 2)
                ml_down_prob = round(100.0 - ml_up_prob, 2)
                
                if signal == 99:
                    signal_text = "⚪ HOLD (รอ)"
                else:
                    signal_text = SignalGenerator.signal_to_text(signal)
            else:
                if not quiet: print("   ❌ Not enough data for lookback")
        
        if not quiet:
            print(f"   💰 Price:  {current_price:,.2f}")
            print(f"   📈 Trend:  {'🟢 UPTREND' if trend == 'UPTREND' else '🔴 DOWNTREND'}")
            print(f"   🎯 Signal: {signal_text}")
            print(f"   📅 Date:   {current_date}")
            print(f"   📊 Trend Stats: Up {summary['uptrend_pct']}% ({summary['uptrend_days']}d) / Down {100-summary['uptrend_pct']:.1f}% ({summary['downtrend_days']}d)")

        results.append({
            'market': market,
            'regime_method': regime_method,
            'uptrend_model': up_model,
            'downtrend_model': down_model,
            'strategy': strategy,
            'trend': trend,
            'signal': signal,
            'signal_text': "HOLD (รอ)" if signal == 99 else signal_text,
            'price': current_price,
            'date': current_date,
            'trend_stats': summary,
            'ml_up_prob': ml_up_prob,
            'ml_down_prob': ml_down_prob,
        })

    return results


def print_summary_table(results):
    """แสดงตารางสรุปสัญญาณ"""
    print("\n" + "=" * 80)
    print("  📋 SUMMARY — สรุปสัญญาณเทรดปัจจุบัน")
    print("=" * 80)
    print(f"  {'Market':<8} {'Price':>12} {'Trend':<12} {'Signal':<12} {'Regime':<15} {'Models':<15}")
    print(f"  {'-'*76}")

    for r in results:
        if r['price'] == 'N/A':
            print(f"  {r['market']:<8} {'N/A':>12} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<15}")
        else:
            trend_icon = '🟢 Up' if r['trend'] == 'UPTREND' else '🔴 Down'
            models = f"↑{r.get('uptrend_model','?')}/↓{r.get('downtrend_model','?')}"
            print(f"  {r['market']:<8} {r['price']:>12,.2f} {trend_icon:<12} {r['signal_text']:<12} {r.get('regime_method',''):<15} {models:<15}")

    print("=" * 80)


# ═════════════════════════════════════════════════════════════════
# 5. PLOT & BACKTEST — สร้างกราฟ backtest
# ═════════════════════════════════════════════════════════════════

def plot_backtest(dates, equity_curve, prices, signals, market, trend, model_name, final_return, bnh_return, regime=None, save_dir='workflows/plot', **kwargs):
    """Generate and save backtest plot with Entry/Exit markers and Regime Background."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
        
    # Align data
    # dates, prices, signals are all length N
    # equity_curve length is N-1 (starts from step 1)
    
    plot_dates = dates[1:]
    plot_prices = prices[1:]
    plot_bnh = (prices[1:] / prices[0]) * INITIAL_CAPITAL
    plot_equity = equity_curve
    plot_signals = signals[:-1] # shift to align with decision points
    
    # Create Trade Markers
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    # 1 = Buy/Hold Long, 0 = Sell/Short, 99 = Neutral/Cash
    prev_pos = 0 
    
    for i in range(len(plot_dates)):
        sig = plot_signals[i]
        curr_pos = 0
        
        if sig == 1: curr_pos = 1
        elif sig == 0: curr_pos = -1
        elif sig == 99: curr_pos = 0
        
        if curr_pos != prev_pos:
            if curr_pos == 1:
                buy_dates.append(plot_dates[i])
                buy_prices.append(plot_prices[i])
            elif curr_pos == -1 or curr_pos == 0:
                 sell_dates.append(plot_dates[i])
                 sell_prices.append(plot_prices[i])
        
        prev_pos = curr_pos

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # 1. Price Chart with Markers
    ax1.plot(plot_dates, plot_prices, label='Price', color='black', alpha=0.6)
    ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy/Long', zorder=5)
    ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell/Exit', zorder=5)
    ax1.set_title(f"{market} {trend} - Price & Signals")
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Regime Background
    if regime is not None and len(regime) == len(dates):
        plot_regime = regime[1:]
        start_idx = 0
        current_val = plot_regime[0]
        
        for i in range(1, len(plot_regime)):
            if plot_regime[i] != current_val:
                color = 'green' if current_val == 1 else 'red'
                if start_idx < len(plot_dates) and i < len(plot_dates):
                    ax1.axvspan(plot_dates[start_idx], plot_dates[i], color=color, alpha=0.1)
                    ax2.axvspan(plot_dates[start_idx], plot_dates[i], color=color, alpha=0.1)
                start_idx = i
                current_val = plot_regime[i]
                
        # Last segment
        color = 'green' if current_val == 1 else 'red'
        if start_idx < len(plot_dates):
            ax1.axvspan(plot_dates[start_idx], plot_dates[-1], color=color, alpha=0.1)
            ax2.axvspan(plot_dates[start_idx], plot_dates[-1], color=color, alpha=0.1)

    # 2. Equity Curve
    ax2.plot(plot_dates, plot_bnh, label=f'Buy & Hold ({bnh_return:.2f}%)', color='gray', linestyle='--')
    ax2.plot(plot_dates, plot_equity, label=f'Strategy ({final_return:.2f}%)', color='blue', linewidth=2)
    ax2.set_title(f"Equity Curve ({model_name})")
    ax2.set_ylabel('Equity ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f"{market}_{trend}_{model_name}.png")
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  📊 Plot saved: {filename}")


def _extract_trade_points(signals, prices, stop_loss_pct=0.05, trailing_stop_pct=0.0, long_only=True, strategy_mode='active'):
    """
    จำลอง simulation เพื่อดึงจุดเข้า-ออกจริง (รวม trailing stop / stop loss exits)
    Returns: list of (index, 'buy'|'sell')
    """
    trade_points = []
    position = 0
    entry_price = 0.0
    highest_price = 0.0

    for i in range(len(signals) - 1):
        signal = signals[i]
        price = prices[i]

        # Track highest price for trailing stop
        if position == 1 and price > highest_price:
            highest_price = price

        # Check exits: trailing stop & stop loss
        if position != 0 and entry_price > 0:
            if position == 1:
                pnl_pct = (price - entry_price) / entry_price
                if trailing_stop_pct > 0:
                    dd_from_peak = (highest_price - price) / highest_price
                    if dd_from_peak > trailing_stop_pct:
                        trade_points.append((i, 'sell'))
                        position = 0
                        entry_price = 0.0
                        highest_price = 0.0
                        continue
            else:
                pnl_pct = (entry_price - price) / entry_price

            if pnl_pct < -stop_loss_pct:
                trade_points.append((i, 'sell'))
                position = 0
                entry_price = 0.0
                highest_price = 0.0
                continue

        # Determine target position
        target_pos = 0
        if signal == 1:
            target_pos = 1
        elif signal == 0:
            target_pos = -1 if not long_only else 0
        elif signal == 99:
            target_pos = position if strategy_mode == 'smart_hold' else 0

        # Execute trade
        if position != target_pos:
            if position != 0:
                trade_points.append((i, 'sell'))
            if target_pos != 0:
                trade_points.append((i, 'buy' if target_pos == 1 else 'sell'))
                entry_price = price
                if target_pos == 1:
                    highest_price = price
            else:
                entry_price = 0.0
                highest_price = 0.0
            position = target_pos

    return trade_points


def run_backtest(markets=None, period='2y'):
    """
    รัน backtest — เหมือน backtest_all_models.py main_combined()
    ใช้ trained ML models จริง + Grid Search หา params ที่ดีที่สุด
    period: '2y' = 2 ปีล่าสุด, 'max' = ทุกข้อมูลตั้งแต่มีจนปัจจุบัน
    """
    import joblib
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model as keras_load_model

    if markets is None:
        markets = MARKETS

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    model_base_dir = os.path.join(project_root, 'all_trad', 'separate_models')
    plot_suffix = '_all' if period == 'max' else ''
    save_dir = os.path.join(script_dir, f'plot{plot_suffix}')

    # Import from backtest_all_models
    sys.path.insert(0, os.path.join(project_root, 'model'))
    try:
        from backtest_all_models import calculate_features, get_selected_features, run_simulation_moo
    except ImportError:
        print("❌ Cannot import from backtest_all_models.py")
        return

    results = []

    for market in markets:
        regime_method = MARKET_REGIME_METHOD.get(market, 'HMM')
        up_model_name = MARKET_UPTREND_MODEL.get(market, '?')
        down_model_name = MARKET_DOWNTREND_MODEL.get(market, '?')

        print(f"\n{'='*60}")
        print(f">> BACKTEST: {market}")
        print(f"   Regime: {regime_method} | Models: ↑{up_model_name} ↓{down_model_name}")
        print("=" * 60)

        # ── 1. Download 2y Data ──
        ticker = TICKERS.get(market)
        if not ticker:
            continue
        print(f"   Loading data for {ticker}...")
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                print("   Empty data. Skipping.")
                continue
            df = calculate_features(df)
            df = df.dropna()
        except Exception as e:
            print(f"   Error loading data: {e}")
            continue

        # ── 2. Regime Detection ──
        print(f"   Detecting regime ({regime_method})...")
        regime = TrendDetector.detect(df, method=regime_method, market_name=market)
        up_days = int(np.sum(regime == 1))
        down_days = int(np.sum(regime == 0))
        print(f"   Regime: {up_days} uptrend / {down_days} downtrend days")

        # ── 3. Load Both Models ──
        uptrend_dir = os.path.join(model_base_dir, f"model_{market}_uptrend")
        downtrend_dir = os.path.join(model_base_dir, f"model_{market}_downtrend")

        try:
            up_model, up_scaler, up_le, up_type = _load_model_and_objects(uptrend_dir, up_model_name, keras_load_model, joblib)
            down_model, down_scaler, down_le, down_type = _load_model_and_objects(downtrend_dir, down_model_name, keras_load_model, joblib)
        except Exception as e:
            print(f"   Error loading models: {e}")
            continue

        if not up_model or not down_model:
            print("   Model loading failed. Skipping.")
            continue

        print(f"   ✅ Models loaded: ↑{up_model_name}({up_type}) ↓{down_model_name}({down_type})")

        # ── 4. Prepare Features ──
        selected_cols = get_selected_features(df)
        if not selected_cols:
            print("   No features found. Skipping.")
            continue

        missing = [f for f in selected_cols if f not in df.columns]
        for m in missing:
            df[m] = 0

        feature_data = df[selected_cols].values

        check_lookback = LOOKBACK
        if up_type == 'ml' and hasattr(up_model, 'n_features_in_'):
            n_expected = up_model.n_features_in_
            n_feats = len(selected_cols)
            if n_feats > 0:
                calculated_lb = n_expected // n_feats
                if calculated_lb != LOOKBACK:
                    check_lookback = calculated_lb

        if len(feature_data) <= check_lookback:
            print("   Not enough data. Skipping.")
            continue

        # ── 5. Scale & Create Windows ──
        scaler = up_scaler or down_scaler
        if scaler:
            try:
                feature_data_scaled = scaler.transform(feature_data)
            except:
                feature_data_scaled = feature_data
        else:
            feature_data_scaled = feature_data

        X_windows_up = []
        X_windows_down = []

        for i in range(check_lookback, len(feature_data_scaled)):
            window = feature_data_scaled[i - check_lookback:i]
            X_windows_up.append(window.flatten() if up_type == 'ml' else window)
            X_windows_down.append(window.flatten() if down_type == 'ml' else window)

        X_up = np.array(X_windows_up)
        X_down = np.array(X_windows_down)

        # ── 6. Predictions ──
        print("   Generating predictions...")
        try:
            if up_type == 'keras':
                probs_up = up_model.predict(X_up, verbose=0)
            else:
                if hasattr(up_model, 'predict_proba'):
                    probs_up = up_model.predict_proba(X_up)
                else:
                    preds = up_model.predict(X_up)
                    probs_up = np.zeros((len(preds), 2))
                    for i, p in enumerate(preds):
                        probs_up[i, int(p)] = 1.0

            if down_type == 'keras':
                probs_down = down_model.predict(X_down, verbose=0)
            else:
                if hasattr(down_model, 'predict_proba'):
                    probs_down = down_model.predict_proba(X_down)
                else:
                    preds = down_model.predict(X_down)
                    probs_down = np.zeros((len(preds), 2))
                    for i, p in enumerate(preds):
                        probs_down[i, int(p)] = 1.0
        except Exception as e:
            print(f"   Prediction error: {e}")
            continue

        # ── 7. Generate Signals & Run Simulation ──
        valid_indices = list(range(check_lookback, len(feature_data)))
        regime_aligned = regime[valid_indices]
        prices = df['Close'].iloc[valid_indices].values
        dates = df.index[valid_indices]
        bnh_return = (prices[-1] - prices[0]) / prices[0] * 100

        print(f"   Generating signals using MOO-optimized parameters...")
        generator = SignalGenerator(market=market)
        best_signals = generator.generate_from_regime(probs_up, probs_down, regime_aligned)

        # Extract confidences for position sizing
        confidences = []
        for i in range(len(regime_aligned)):
            if regime_aligned[i] == 1:
                conf = np.max(probs_up[i])
            else:
                conf = np.max(probs_down[i])
            confidences.append(conf)
        confidences = np.array(confidences)

        print(f"   Running simulation...")
        engine = TradingEngine(market=market, initial_capital=INITIAL_CAPITAL)
        result = engine.run(best_signals, prices, confidences=confidences, regimes=regime_aligned)

        best_eq = result['equity_curve']
        best_trades = result['n_trades']
        best_winrate = result['win_rate']
        best_dd = result['max_drawdown']
        
        market_params = OPTIMIZED_PARAMS.get(market, {})
        best_params = {
            'confidence': generator.confidence_threshold, 
            'long': generator.long_threshold, 
            'short': generator.short_threshold,
            'stop_loss': market_params.get('stop_loss', 0.05),
            'strategy': MARKET_STRATEGIES.get(market, 'active'),
            'trailing': MARKET_TRAILING_STOPS.get(market, 0.0), 
            'long_only': MARKET_LONG_ONLY.get(market, False),
            'fallback': generator.fallback
        }

        print(f"   📊 Loaded MOO Params: {best_params}")

        strategy_return = (best_eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        vs_bnh = strategy_return - bnh_return

        print(f"   📈 Strategy Return: {strategy_return:+.2f}%")
        print(f"   📊 Buy & Hold:      {bnh_return:+.2f}%")
        print(f"   📉 Max Drawdown:    {best_dd:.2f}%")
        print(f"   🔄 Trades:          {best_trades}")
        print(f"   ✅ Win Rate:        {best_winrate:.1f}%")
        print(f"   💰 Final Equity:    ${best_eq[-1]:,.2f}")
        print(f"   {'🟢' if vs_bnh >= 0 else '🔴'} vs B&H: {vs_bnh:+.2f}%")

        # ── 8. Export History Dump ──
        # Reconstruct positions from trade points
        trade_events = _extract_trade_points(
            best_signals, prices,
            stop_loss_pct=best_params.get('stop_loss', 0.05),
            trailing_stop_pct=best_params.get('trailing', 0.0),
            long_only=best_params.get('long_only', True),
            strategy_mode=best_params.get('strategy', 'active')
        )
        event_dict = {idx: action for idx, action in trade_events}
        current_pos = 0
        bnh_curve = (prices / prices[0]) * INITIAL_CAPITAL
        
        # Align equity curve to be length N (same as dates and prices)
        # best_eq is length N-1, representing the equity at the end of each day starting from day 1
        aligned_eq = [INITIAL_CAPITAL] + list(best_eq)
        
        market_history = []
        for i in range(len(dates)):
            # Update Position first if there was an action ON this index
            if i in event_dict:
                current_pos = 1 if event_dict[i] == 'buy' else 0

            sig = best_signals[i]
            sg_text = "⚪ HOLD (รอ)"
            if sig == 1: sg_text = "🟢 BUY"
            elif sig == 0: sg_text = "🔴 SELL"
                
            reg_val = regime_aligned[i]
            trend_str = "1 (Uptrend)" if reg_val == 1 else "0 (Downtrend)"
            
            # Extract ML continuous probability
            if reg_val == 1:
                p_up = probs_up[i, 1] if probs_up.shape[1] > 1 else probs_up[i, 0]
                p_dn = probs_up[i, 0] if probs_up.shape[1] > 1 else 1-probs_up[i, 0]
            else:
                p_up = probs_down[i, 1] if probs_down.shape[1] > 1 else probs_down[i, 0]
                p_dn = probs_down[i, 0] if probs_down.shape[1] > 1 else 1-probs_down[i, 0]
                
            market_history.append({
                'date': dates[i].strftime('%Y-%m-%d'),
                'market': market,
                'price': float(prices[i]),
                'trend_regime': trend_str,
                'ml_up_prob': round(float(p_up) * 100, 2),
                'ml_down_prob': round(float(p_dn) * 100, 2),
                'signal_action': sg_text,
                'position': float(current_pos),
                'equity_curve': float(aligned_eq[i]),
                'bnh_curve': float(bnh_curve[i])
            })
            
        history_file = os.path.join(save_dir, f'{market}_history.json')
        import json
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(market_history, f, indent=2, ensure_ascii=False)
            
        # ── 8.5. Export Plot Data Dump ──
        plot_data = {
            "dates": [d.strftime('%Y-%m-%d') for d in dates],
            "prices": prices.tolist(),
            "bnh_return": bnh_return,
            "strategy_return": strategy_return,
            "equity_curve": best_eq,
            "signals": best_signals.tolist(),
            "regime": regime_aligned.tolist(),
            "sim_params": best_params,
            "market": market
        }
        plot_data_file = os.path.join(save_dir, f'{market}_plot_data.json')
        with open(plot_data_file, 'w', encoding='utf-8') as f:
            json.dump(plot_data, f, indent=2, ensure_ascii=False)
            
        # ── 8.6. Auto-Ingest into SQLite Database ──
        db_path = os.path.join(script_dir, 'trading_database.sqlite')
        try:
            import sqlite3
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Setup Tables (in case llm_agent hasn't run yet)
                table_name = f"signals_history_{market}"
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS "{table_name}" (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        market TEXT,
                        price REAL,
                        trend_regime TEXT,
                        ml_up_prob REAL,
                        ml_down_prob REAL,
                        signal_action TEXT,
                        position REAL,
                        equity_curve REAL,
                        bnh_curve REAL,
                        UNIQUE(date)
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        market TEXT PRIMARY KEY,
                        base_return_pct REAL,
                        bnh_return_pct REAL,
                        win_rate_pct REAL,
                        total_trades INTEGER,
                        max_drawdown_pct REAL,
                        updated_at TEXT
                    )
                ''')

                # Clear old data 
                cursor.execute(f'DELETE FROM "{table_name}"')
                
                # Insert granular daily signals
                for row in market_history:
                    cursor.execute(f'''
                        INSERT INTO "{table_name}" 
                        (date, market, price, trend_regime, ml_up_prob, ml_down_prob, signal_action, position, equity_curve, bnh_curve)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row['date'], row['market'], row['price'], row['trend_regime'], 
                          row['ml_up_prob'], row['ml_down_prob'], row['signal_action'], 
                          row['position'], row['equity_curve'], row['bnh_curve']))
                
                # Insert strategy performance summary
                from datetime import datetime
                cursor.execute('''
                    INSERT OR REPLACE INTO strategy_performance 
                    (market, base_return_pct, bnh_return_pct, win_rate_pct, total_trades, max_drawdown_pct, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (market, strategy_return, bnh_return, best_winrate, best_trades, best_dd, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
                conn.commit()
                conn.close()
                print(f"   📥 Auto-ingested {len(market_history)} days and performance metrics into Database")
        except Exception as e:
            print(f"   ⚠️ Database auto-ingestion failed: {e}")

        # ── 9. Plot ──
        plot_backtest(
            dates=dates,
            equity_curve=best_eq,
            prices=prices,
            signals=best_signals,
            market=market,
            trend="Combined",
            model_name="Hybrid_MOO",
            final_return=strategy_return,
            bnh_return=bnh_return,
            regime=regime_aligned,
            save_dir=save_dir
        )

        results.append({
            'Market': market,
            'Regime Method': regime_method,
            'Uptrend Model': up_model_name,
            'Downtrend Model': down_model_name,
            'Type': f"MOO ({best_params.get('strategy', 'grid')})",
            'Return (%)': round(strategy_return, 2),
            'B&H (%)': round(bnh_return, 2),
            'Max DD (%)': round(best_dd, 2),
            'Trades': best_trades,
            'Final $': round(best_eq[-1], 2),
            'Confidence': best_params.get('confidence', 0),
            'Threshold Long': best_params.get('long', 0),
            'Threshold Short': best_params.get('short', 0),
            'Stop Loss': best_params.get('stop_loss', 0),
            'Win Rate (%)': round(best_winrate, 1),
        })

        tf.keras.backend.clear_session()

    # Print Summary
    if results:
        period_label = 'ALL DATA' if period == 'max' else '2 Years'
        print("\n" + "=" * 80)
        print(f"  📋 BACKTEST SUMMARY ({period_label}) — Grid Search Optimized")
        print("=" * 80)
        print(f"  {'Market':<8} {'Return':>10} {'B&H':>10} {'vs B&H':>10} {'DD':>8} {'Trades':>8} {'WR':>6} {'Final $':>12}")
        print(f"  {'-'*78}")
        for r in results:
            vs = r['Return (%)'] - r['B&H (%)']
            vs_icon = '🟢' if vs >= 0 else '🔴'
            print(f"  {r['Market']:<8} {r['Return (%)']:>+10.2f} {r['B&H (%)']:>10.2f} {vs_icon}{vs:>+8.2f} {r['Max DD (%)']:>8.2f} {r['Trades']:>8d} {r['Win Rate (%)']:>5.1f}% {r['Final $']:>12,.2f}")
        print("=" * 80)
        print(f"\n  📁 Plots saved to: {save_dir}/")

        csv_path = os.path.join(save_dir, 'backtest_results.csv')
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"  📄 Results saved to: {csv_path}")


def _load_model_and_objects(model_dir, model_name, keras_load_model, joblib):
    """Load model, scaler, label_encoder (เหมือน backtest_all_models.py)"""
    model = None
    scaler = None
    le = None
    model_type = 'ml'

    p_keras = os.path.join(model_dir, f"{model_name}.keras")
    p_pkl = os.path.join(model_dir, f"{model_name}.pkl")

    if os.path.exists(p_keras):
        model = keras_load_model(p_keras)
        model_type = 'keras'
    elif os.path.exists(p_pkl):
        model = joblib.load(p_pkl)
        model_type = 'ml'

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(le_path):
        le = joblib.load(le_path)

    return model, scaler, le, model_type


def _calculate_drawdown_val(equity_curve):
    """คำนวณ Maximum Drawdown จาก equity list"""
    if not equity_curve:
        return 0.0
    eq = pd.Series(equity_curve)
    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    return drawdown.min() * 100


def save_signals_to_db(results, quiet=False):
    """
    Save the daily generated signals to the SQLite database.
    Prevents duplicates by checking the current date.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, 'trading_database.sqlite')
        
        today_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        saved_count = 0
        
        for m in results:
            market = m['market']
            table_name = f"signals_history_{market}"
            
            c.execute(f'''
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    market TEXT,
                    price REAL,
                    trend_regime TEXT,
                    ml_up_prob REAL,
                    ml_down_prob REAL,
                    signal_action TEXT,
                    position REAL,
                    equity_curve REAL,
                    bnh_curve REAL,
                    UNIQUE(date)
                )
            ''')
            
            # Check if we already have today's insert
            c.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE date=?', (today_date,))
            if c.fetchone()[0] == 0:
                trend_str = "1 (Uptrend)" if m['trend'] == 'UPTREND' else "0 (Downtrend)"
                price = m.get('price', 0)
                if price == 'N/A': price = 0
                
                # Use real ML probabilities from signal generation
                up_prob = m.get('ml_up_prob', 0.0)
                down_prob = m.get('ml_down_prob', 0.0)
                
                # Determine position from signal (1=long, -1=short, 0=cash)
                sig = m.get('signal', 99)
                position = 1.0 if sig == 1 else (-1.0 if sig == 0 else 0.0)
                
                # Chain equity/bnh from last known row
                equity_val = 0.0
                bnh_val = 0.0
                try:
                    c.execute(f'SELECT equity_curve, bnh_curve, price FROM "{table_name}" WHERE equity_curve != 0 ORDER BY date DESC LIMIT 1')
                    last_row = c.fetchone()
                    if last_row and last_row[2] > 0:
                        prev_eq, prev_bnh, prev_price = last_row
                        price_ratio = price / prev_price if prev_price > 0 else 1.0
                        # B&H always tracks price
                        bnh_val = round(prev_bnh * price_ratio, 4)
                        # Equity tracks based on position
                        if position == 1.0:
                            equity_val = round(prev_eq * price_ratio, 4)
                        elif position == -1.0:
                            equity_val = round(prev_eq * (2 - price_ratio), 4)
                        else:
                            equity_val = prev_eq  # Cash: no change
                except Exception:
                    pass
                
                c.execute(f'''
                    INSERT INTO "{table_name}" 
                    (date, market, price, trend_regime, ml_up_prob, ml_down_prob, signal_action, position, equity_curve, bnh_curve)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (today_date, market, price, trend_str, up_prob, down_prob, m['signal_text'], position, equity_val, bnh_val))
                saved_count += 1
                
        conn.commit()
        conn.close()
        
        if not quiet and saved_count > 0:
            print(f"\n  ✅ Successfully archived {saved_count} daily signals to SQLite Database.")
        elif not quiet and saved_count == 0:
            print(f"\n  ℹ️ Database up to date. No new records inserted for {today_date}.")
            
    except Exception as e:
        if not quiet:
            print(f"\n  ❌ Failed to save signals to Database: {e}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Trading System — ระบบเทรดอัตโนมัติ')
    parser.add_argument('--market', type=str, default=None,
                        help='ตลาดที่ต้องการดู (US, UK, Thai, Gold, BTC)')
    parser.add_argument('--backtest', action='store_true',
                        help='รัน backtest 2 ปีล่าสุด พร้อมสร้างกราฟ')
    parser.add_argument('--backtest-all', action='store_true',
                        help='รัน backtest ทุกข้อมูลตั้งแต่มีจนปัจจุบัน พร้อมสร้างกราฟ')
    parser.add_argument('--json', action='store_true',
                        help='แสดงผลลัพธ์เป็น JSON สำหรับให้ LLM หรือโปรแกรมอื่นนำไปใช้งานต่อ')

    args = parser.parse_args()

    if args.market:
        markets = [args.market]
    else:
        markets = MARKETS

    if args.json:
        # JSON Output Mode for LLMs / Integrations
        # Suppress ALL stdout noise during JSON mode
        import json
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Capture any stray prints
        try:
            results = get_current_signals(markets=markets, quiet=True)
        finally:
            sys.stdout = old_stdout  # Restore stdout
            
        import numpy as np
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
                
        print(json.dumps(results, indent=2, ensure_ascii=False, cls=NpEncoder))
        sys.exit(0)

    print("=" * 80)
    print("  🚀 TRADING SYSTEM — ระบบเทรดอัตโนมัติ")
    print("  📡 Auto Regime: ใช้วิธีที่ดีที่สุดต่อตลาด (จาก Backtest Results)")
    print("=" * 80)

    if args.backtest_all:
        run_backtest(markets=markets, period='max')
    elif args.backtest:
        run_backtest(markets=markets, period='2y')
    else:
        # Show current signals
        results = get_current_signals(markets=markets)
        print_summary_table(results)

        # Show parameters
        print("\n📋 MOO-Optimized Parameters:")
        for m in markets:
            sg = SignalGenerator(market=m)
            te = TradingEngine(market=m)
            regime = MARKET_REGIME_METHOD.get(m, 'HMM')
            up_m = MARKET_UPTREND_MODEL.get(m, '?')
            down_m = MARKET_DOWNTREND_MODEL.get(m, '?')
            print(f"  {m:>5}: {sg.get_params_text()}")
            print(f"         {te.get_config_text()}")
            print(f"         Regime: {regime} | Models: ↑{up_m} / ↓{down_m}")

        # Auto-Save to SQLite Database
        save_signals_to_db(results)

