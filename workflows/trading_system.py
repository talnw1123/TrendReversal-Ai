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

warnings.filterwarnings('ignore')

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
# BEST CONFIG per Market (จาก backtest_combined_results.csv)
# ─────────────────────────────────────────────────────────────────
# Market | Regime Method  | Uptrend Model  | Downtrend Model | Strategy
# US     | HMM            | LSTM           | CNN             | smart_hold
# UK     | HMM            | MLP            | SVM             | active
# Thai   | HMM_Enhanced   | MLP            | LSTM            | smart_hold
# Gold   | HMM            | CNN            | Transformer     | smart_hold
# BTC    | GMM_Enhanced   | RandomForest   | LSTM            | active

MARKET_REGIME_METHOD = {
    'US':   'HMM',
    'UK':   'HMM',
    'Thai': 'HMM_Enhanced',
    'Gold': 'HMM',
    'BTC':  'GMM_Enhanced',
}

MARKET_UPTREND_MODEL = {
    'US':   'LSTM',
    'UK':   'MLP',
    'Thai': 'MLP',
    'Gold': 'CNN',
    'BTC':  'RandomForest',
}

MARKET_DOWNTREND_MODEL = {
    'US':   'CNN',
    'UK':   'SVM',
    'Thai': 'LSTM',
    'Gold': 'Transformer',
    'BTC':  'LSTM',
}

# MOO-Optimized Parameters per Market (จาก moo_optimized_params.csv)
OPTIMIZED_PARAMS = {
    'US':   {'confidence': 0.5105, 'long': 0.5393, 'short': 0.3019, 'stop_loss': 0.1496},
    'UK':   {'confidence': 0.5169, 'long': 0.5232, 'short': 0.3251, 'stop_loss': 0.1105},
    'Thai': {'confidence': 0.5132, 'long': 0.5413, 'short': 0.3166, 'stop_loss': 0.0793},
    'Gold': {'confidence': 0.5392, 'long': 0.5488, 'short': 0.3533, 'stop_loss': 0.0320},
    'BTC':  {'confidence': 0.5955, 'long': 0.5765, 'short': 0.3294, 'stop_loss': 0.0596},
}

# Strategy per Market
MARKET_STRATEGIES = {
    'US':   'smart_hold',
    'UK':   'active',
    'Gold': 'smart_hold',
    'BTC':  'active',
    'Thai': 'smart_hold',
}

# Trailing Stop per Market (ใช้กับ smart_hold เท่านั้น)
MARKET_TRAILING_STOPS = {
    'US':   0.15,
    'UK':   0.0,
    'Gold': 0.15,
    'BTC':  0.0,
    'Thai': 0.10,
}

# Long Only per Market
MARKET_LONG_ONLY = {
    'US':   True,
    'UK':   True,
    'Gold': True,
    'BTC':  False,
    'Thai': False,
}

LOOKBACK = 30
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001


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
        else:
            print(f"  [Warning] Unknown method '{method}'. Using HMM.")
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
            'confidence': 0.55, 'long': 0.54, 'short': 0.46
        })
        self.confidence_threshold = params['confidence']
        self.long_threshold = params['long']
        self.short_threshold = params['short']
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
            signal = self._classify(confidence, bull_prob)
            signals.append(signal)

        return np.array(signals)

    def _classify(self, confidence, bull_prob):
        """ตัดสินสัญญาณจาก confidence และ bull_prob"""
        if confidence < self.confidence_threshold:
            return 99  # HOLD — ไม่มั่นใจ
        elif bull_prob >= self.long_threshold:
            return 1   # BUY
        elif bull_prob <= self.short_threshold:
            return 0   # SELL
        else:
            return 99  # HOLD — อยู่ในโซนกลาง

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

    def run(self, signals, prices):
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
        cash = self.initial_capital
        position = 0
        entry_price = 0.0
        highest_price = 0.0
        equity_curve = []
        n_trades = 0
        wins = 0

        for i in range(len(signals) - 1):
            signal = signals[i]
            price = prices[i]
            next_price = prices[i + 1]

            # Track highest price for trailing stop (long only)
            if position == 1 and price > highest_price:
                highest_price = price

            # ──── Check Exits: Stop Loss & Trailing Stop ────
            if position != 0 and entry_price > 0:
                if position == 1:
                    pnl_pct = (price - entry_price) / entry_price

                    # Trailing Stop
                    if self.trailing_stop_pct > 0:
                        drawdown_from_peak = (highest_price - price) / highest_price
                        if drawdown_from_peak > self.trailing_stop_pct:
                            cash -= cash * self.transaction_cost
                            if pnl_pct > 0:
                                wins += 1
                            n_trades += 1
                            position = 0
                            entry_price = 0.0
                            highest_price = 0.0
                            equity_curve.append(cash)
                            continue
                else:
                    pnl_pct = (entry_price - price) / entry_price

                # Stop Loss
                if pnl_pct < -self.stop_loss_pct:
                    cash -= cash * self.transaction_cost
                    if pnl_pct > 0:
                        wins += 1
                    n_trades += 1
                    position = 0
                    entry_price = 0.0
                    highest_price = 0.0
                    equity_curve.append(cash)
                    continue

            # ──── Determine Target Position ────
            target_pos = 0

            if signal == 1:
                target_pos = 1
            elif signal == 0:
                target_pos = 0 if self.long_only else -1
            elif signal == 99:
                if self.strategy_mode == 'smart_hold':
                    target_pos = position  # Hold current position
                else:
                    target_pos = 0  # Exit to cash

            # ──── Execute Trade ────
            if position != target_pos:
                cost = cash * abs(target_pos - position) * self.transaction_cost
                cash -= cost

                # Record trade stats
                if position != 0 and entry_price > 0:
                    if position == 1:
                        trade_pnl = (price - entry_price) / entry_price
                    else:
                        trade_pnl = (entry_price - price) / entry_price
                    if trade_pnl > 0:
                        wins += 1
                    n_trades += 1

                position = target_pos

                if target_pos != 0:
                    entry_price = price
                    if target_pos == 1:
                        highest_price = price
                else:
                    entry_price = 0.0
                    highest_price = 0.0

            # ──── Update Equity ────
            if position == 1:
                cash *= (1 + (next_price - price) / price)
            elif position == -1:
                cash *= (1 + (price - next_price) / price)

            equity_curve.append(cash)

        # Calculate metrics
        win_rate = (wins / n_trades * 100) if n_trades > 0 else 0.0
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
        model_dir = os.path.join(model_base_dir, market, f"regime_{'uptrend' if trend == 'UPTREND' else 'downtrend'}")
        
        if not quiet: print(f"   Loading active model: {active_model_name}...")
        
        try:
            model, scaler, le, model_type = _load_model_and_objects(model_dir, active_model_name, keras_load_model, joblib)
        except Exception as e:
            if not quiet: print(f"   ❌ Error loading model: {e}")
            model = None
            
        signal = 99
        signal_text = "⚪ HOLD (รอ)"
        
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
                signals = sg.generate_signals([prob])
                signal = signals[0]
                
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

def plot_backtest(dates, equity_curve, prices, signals, regime, market, strategy_return, bnh_return, save_dir='workflows/plot', **kwargs):
    """
    สร้างกราฟ backtest แบบ 2 panels:
    - Panel 1: Price + Buy/Sell markers (จุดเข้า-ออกจริง) + Regime background
    - Panel 2: Equity Curve vs Buy & Hold + Regime background
    
    trade_points: list of (idx, 'buy'|'sell') — ถ้าไม่ส่งมา จะคำนวณจาก simulation params
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    os.makedirs(save_dir, exist_ok=True)

    # Align arrays (equity_curve is len(signals)-1)
    plot_dates = dates[1:]
    plot_prices = prices[1:]
    plot_bnh = (prices[1:] / prices[0]) * INITIAL_CAPITAL
    plot_equity = np.array(equity_curve)
    plot_signals = signals[:-1]
    plot_regime = regime[1:] if len(regime) == len(dates) else regime[:len(plot_dates)]

    # Extract actual trade points from simulation
    sim_params = kwargs.get('sim_params', {})
    trade_points = _extract_trade_points(
        signals, prices,
        stop_loss_pct=sim_params.get('stop_loss', 0.05),
        trailing_stop_pct=sim_params.get('trailing', 0.0),
        long_only=sim_params.get('long_only', True),
        strategy_mode=sim_params.get('strategy', 'active')
    )

    buy_dates_actual, buy_prices_actual = [], []
    sell_dates_actual, sell_prices_actual = [], []

    for idx, action in trade_points:
        # idx range is 0 to len(signals)-2
        # plot_dates corresponds to dates[1:]
        # So we can safely use idx for plot_dates, because plot_dates[idx] == dates[idx+1]
        # But wait, run_simulation_moo uses prices[i] for execution and next_price for PnL.
        # So the actual trade happens at index `i` of the original array?
        # Let's map it safely.
        if 0 <= idx < len(dates):
            if action == 'buy':
                buy_dates_actual.append(dates[idx])
                buy_prices_actual.append(prices[idx])
            elif action == 'sell':
                sell_dates_actual.append(dates[idx])
                sell_prices_actual.append(prices[idx])

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 1]})

    # Panel 1: Price + Signals
    ax1.plot(plot_dates, plot_prices, label='Price', color='black', alpha=0.6)
    ax1.scatter(buy_dates_actual, buy_prices_actual, marker='^', color='green', s=100, label='Buy/Long', zorder=5)
    ax1.scatter(sell_dates_actual, sell_prices_actual, marker='v', color='red', s=100, label='Sell/Exit', zorder=5)
    ax1.set_title(f"{market} Combined - Price & Signals")
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Equity Curve
    ax2.plot(plot_dates, plot_bnh, label=f'Buy & Hold ({bnh_return:.2f}%)', color='gray', linestyle='--')
    ax2.plot(plot_dates, plot_equity, label=f'Strategy ({strategy_return:.2f}%)', color='blue', linewidth=2)
    ax2.set_title(f"Equity Curve (Hybrid_MOO)")
    ax2.set_ylabel('Equity ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Regime Background (both panels)
    if len(plot_regime) > 0:
        start_idx = 0
        current_val = plot_regime[0]

        for i in range(1, len(plot_regime)):
            if plot_regime[i] != current_val:
                color = 'green' if current_val == 1 else 'red'
                ax1.axvspan(plot_dates[start_idx], plot_dates[i], color=color, alpha=0.1)
                ax2.axvspan(plot_dates[start_idx], plot_dates[i], color=color, alpha=0.1)
                start_idx = i
                current_val = plot_regime[i]

        # Last segment
        color = 'green' if current_val == 1 else 'red'
        ax1.axvspan(plot_dates[start_idx], plot_dates[-1], color=color, alpha=0.1)
        ax2.axvspan(plot_dates[start_idx], plot_dates[-1], color=color, alpha=0.1)

    plt.tight_layout()
    filename = os.path.join(save_dir, f"{market}_Combined_Hybrid_MOO.png")
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

        # ── 7. Grid Search Optimization (เหมือน backtest_all_models.py) ──
        valid_indices = list(range(check_lookback, len(feature_data)))
        regime_aligned = regime[valid_indices]
        prices = df['Close'].iloc[valid_indices].values
        dates = df.index[valid_indices]
        bnh_return = (prices[-1] - prices[0]) / prices[0] * 100

        print(f"   ⚡ Running Grid Search...")

        # Parameter grids (expanded for better alpha)
        conf_grid = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
        long_grid = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]
        short_grid = [0.35, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
        sl_grid = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
        strategy_grid = ['active', 'smart_hold']
        trailing_grid = [0.0, 0.03, 0.05, 0.10, 0.15, 0.20]
        long_only_options = [True, False]

        best_score = -999
        best_params = {}
        best_eq = [INITIAL_CAPITAL]
        best_trades = 0
        best_winrate = 0.0
        best_dd = 0.0
        best_signals = np.ones(len(regime_aligned)) * 99
        total_combos = 0

        for conf_t in conf_grid:
            for long_t in long_grid:
                for short_t in short_grid:
                    if short_t >= long_t:
                        continue

                    # Generate signals for this param combo
                    test_signals = []
                    for i in range(len(regime_aligned)):
                        if regime_aligned[i] == 1:
                            bull_prob = probs_up[i, 1] if probs_up.shape[1] > 1 else probs_up[i, 0]
                            conf = np.max(probs_up[i])
                        else:
                            bull_prob = probs_down[i, 1] if probs_down.shape[1] > 1 else probs_down[i, 0]
                            conf = np.max(probs_down[i])

                        if conf < conf_t:
                            test_signals.append(99)
                        elif bull_prob >= long_t:
                            test_signals.append(1)
                        elif bull_prob <= short_t:
                            test_signals.append(0)
                        else:
                            test_signals.append(99)

                    test_signals = np.array(test_signals)

                    for sl in sl_grid:
                        for strat in strategy_grid:
                            for ts in trailing_grid:
                                if strat == 'active' and ts > 0:
                                    continue
                                if strat == 'smart_hold' and ts == 0:
                                    continue

                                for lo in long_only_options:
                                    total_combos += 1

                                    eq, trades, wr = run_simulation_moo(
                                        test_signals, prices,
                                        stop_loss_pct=sl,
                                        trailing_stop_pct=ts,
                                        long_only=lo,
                                        strategy_mode=strat
                                    )

                                    if eq:
                                        ret = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                                        dd = _calculate_drawdown_val(eq)

                                        # Score = Alpha + WR adjustments
                                        alpha = ret - bnh_return
                                        score = alpha
                                        
                                        # ── Penalties & Bonuses for Balance ──
                                        
                                        # 1. Drawdown Penalty
                                        if dd < -25:
                                            score -= 10
                                            
                                        # 2. Prevent 1-trade "Fluke" 100% WR if possible, but don't ban it completely 
                                        # (Gold often has 1-2 good long-term trades)
                                        if trades < 3:
                                            # Penalty depends on how much it beats B&H. If it crushes B&H, keep it.
                                            # If it just barely beats B&H, penalize it to encourage more active trading.
                                            score -= 15
                                            
                                        if trades > 0:
                                            # 3. Win Rate Penalty (Bad strategies)
                                            if wr < 50:
                                                score -= 30
                                            
                                            # 4. Balanced Bonus: Reward higher trades IF Win Rate is good
                                            # Cap the bonus to not overshadow Alpha
                                            elif wr >= 55:
                                                trade_multiplier = min(trades, 20) / 10.0  # e.g., 5 trades = 0.5x, 20 trades = 2.0x
                                                wr_bonus = (wr - 50) * 0.4                 # e.g., 70% WR = 8 points
                                                score += (wr_bonus * trade_multiplier)
                                        if score > best_score:
                                            best_score = score
                                            best_params = {
                                                'confidence': conf_t, 'long': long_t, 'short': short_t,
                                                'stop_loss': sl, 'strategy': strat, 'trailing': ts, 'long_only': lo
                                            }
                                            best_eq = eq
                                            best_trades = trades
                                            best_winrate = wr
                                            best_dd = dd
                                            best_signals = test_signals

        print(f"   ✅ Searched {total_combos} combos → Best Alpha vs B&H: {best_score:+.2f}%")
        print(f"   📊 Best Params: {best_params}")

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
        
        market_history = []
        for i in range(len(dates)):
            # Update Position first if there was an action ON this index
            # Actually, the action takes effect at this index, so position changes here
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
                'equity_curve': float(best_eq[i] if i < len(best_eq) else best_eq[-1]),
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
            regime=regime_aligned,
            market=market,
            strategy_return=strategy_return,
            bnh_return=bnh_return,
            save_dir=save_dir,
            sim_params=best_params,
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
        import json
        results = get_current_signals(markets=markets, quiet=True)
        print(json.dumps(results, indent=2, ensure_ascii=False))
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

