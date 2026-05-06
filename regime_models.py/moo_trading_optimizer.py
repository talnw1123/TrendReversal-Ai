"""
Multi-Objective Optimization for Trading Strategy Parameters
Uses NSGA-II to optimize 4 conflicting objectives:
  1. Maximize Return
  2. Minimize Max Drawdown
  3. Maximize Sharpe Ratio
  4. Maximize Win Rate

Decision Variables:
  - confidence_threshold (0.50 - 0.70)
  - threshold_long (0.52 - 0.70)
  - threshold_short (0.30 - 0.48)
  - stop_loss_pct (0.02 - 0.15)

Usage:
  python moo_trading_optimizer.py                          # Optimize all markets
  python moo_trading_optimizer.py --market US               # Optimize single market
  python moo_trading_optimizer.py --market US --compare     # Compare optimized vs original
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../model'))
sys.path.insert(0, os.path.dirname(__file__))

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.decomposition.asf import ASF

from config import TICKERS, BEST_MODELS, LOOKBACK, BINARY_MODE, TREND_METHOD
from model.features import calculate_features, get_selected_features
from model.regime_detection import RegimeDetector

# ============================================================================
# CONSTANTS
# ============================================================================
INITIAL_CAPITAL = 10000.0

# Original (baseline) parameters
ORIGINAL_PARAMS = {
    'confidence_threshold': 0.55,
    'threshold_long': 0.54,
    'threshold_short': 0.46,
    'stop_loss_pct': 0.05,
}

# ============================================================================
# DATA PREPARATION (cached per market)
# ============================================================================
_data_cache = {}


def prepare_market_data(market: str):
    """Fetch and prepare market data with features and models. Cached."""
    if market in _data_cache:
        return _data_cache[market]

    ticker = TICKERS.get(market)
    if not ticker:
        return None

    print(f"  📥 Downloading {market} ({ticker})...")
    df = yf.download(ticker, period="max", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 500:
        print(f"  ❌ Not enough data for {market}")
        return None

    print(f"  📊 Calculating features ({len(df)} rows)...")
    df_feat = calculate_features(df)
    feature_cols = get_selected_features(df_feat)

    print(f"  🔍 Detecting regimes ({TREND_METHOD})...")
    if TREND_METHOD == 'hmm':
        is_uptrend = RegimeDetector.detect_hmm(df)
    elif TREND_METHOD == 'gmm':
        is_uptrend = RegimeDetector.detect_gmm(df)
    elif TREND_METHOD == 'adx_supertrend':
        is_uptrend = RegimeDetector.detect_adx_supertrend(df)
    else:
        sma = df['Close'].rolling(200).mean()
        is_uptrend = (df['Close'] > sma).astype(int).fillna(0).values

    if isinstance(is_uptrend, pd.Series):
        is_uptrend = is_uptrend.values

    # Load models
    print(f"  🧠 Loading models...")
    import tensorflow as tf
    import joblib

    models = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for trend in ['uptrend', 'downtrend']:
        model_name = BEST_MODELS.get(market, {}).get(trend)
        if not model_name:
            continue

        # Try paths in order: market-specific → general separate_models → model/
        search_dirs = [
            os.path.join(script_dir, f"separate_models/model_{market}_{trend}"),
            os.path.join(script_dir, f"separate_models/model_{trend}"),
            os.path.join(script_dir, f"../model/model_{trend}"),
        ]

        loaded = False
        for model_dir in search_dirs:
            keras_path = os.path.join(model_dir, f"{model_name}.keras")
            pkl_path = os.path.join(model_dir, f"{model_name}.pkl")

            if os.path.exists(keras_path):
                models[trend] = {'model': tf.keras.models.load_model(keras_path), 'name': model_name, 'type': 'keras'}
                print(f"    Loaded {trend} model from: {model_dir}")
                loaded = True
                break
            elif os.path.exists(pkl_path):
                models[trend] = {'model': joblib.load(pkl_path), 'name': model_name, 'type': 'sklearn'}
                print(f"    Loaded {trend} model from: {model_dir}")
                loaded = True
                break

        if not loaded:
            print(f"  ⚠️  Model not found for {trend}: {model_name}")

    if len(models) < 2:
        print(f"  ❌ Need both uptrend and downtrend models")
        return None

    print(f"  ✅ Models loaded: uptrend={models['uptrend']['name']}, downtrend={models['downtrend']['name']}")

    # Precompute signals for all data points (batch mode for speed)
    print(f"  🔮 Precomputing predictions (batch mode)...")
    data = df_feat[feature_cols].values
    start_idx = max(LOOKBACK + 200, 250)
    n_points = len(data) - start_idx

    bull_probs = np.full(len(data), 0.5)  # Default neutral

    # Step 1: Build all sequences and group by trend
    uptrend_indices = []
    uptrend_sequences = []
    downtrend_indices = []
    downtrend_sequences = []

    for i in range(start_idx, len(data)):
        if i < LOOKBACK:
            continue

        sequence = data[i - LOOKBACK:i].copy()
        mean = np.nanmean(sequence, axis=0)
        std = np.nanstd(sequence, axis=0) + 1e-9
        sequence = (sequence - mean) / std
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

        if is_uptrend[i] == 1:
            uptrend_indices.append(i)
            uptrend_sequences.append(sequence)
        else:
            downtrend_indices.append(i)
            downtrend_sequences.append(sequence)

    # Step 2: Batch predict per trend
    for trend, indices, sequences in [
        ('uptrend', uptrend_indices, uptrend_sequences),
        ('downtrend', downtrend_indices, downtrend_sequences),
    ]:
        if not sequences:
            continue

        m_info = models.get(trend)
        if not m_info:
            continue

        X_batch = np.array(sequences)  # (N, LOOKBACK, features)
        model = m_info['model']
        print(f"    Predicting {trend}: {len(X_batch)} samples...")

        try:
            if m_info['type'] == 'sklearn':
                X_flat = X_batch.reshape(len(X_batch), -1)
                probs = model.predict_proba(X_flat)
            else:
                probs = model.predict(X_batch, verbose=0, batch_size=512)

            for j, idx in enumerate(indices):
                bull_probs[idx] = probs[j, 1]  # Probability of bullish
        except Exception as e:
            print(f"    ⚠️ Prediction error for {trend}: {e}")

    print(f"  ✅ Predictions complete")

    # Returns for vectorized backtest
    prices = df_feat['Close'].values
    returns = np.zeros(len(prices))
    returns[1:] = np.diff(prices) / prices[:-1]

    cache_entry = {
        'df': df_feat,
        'prices': prices,
        'returns': returns,
        'is_uptrend': is_uptrend,
        'bull_probs': bull_probs,
        'start_idx': start_idx,
        'feature_cols': feature_cols,
    }
    _data_cache[market] = cache_entry
    print(f"  ✅ {market} data prepared ({n_points} trading days)")
    return cache_entry


# ============================================================================
# FAST VECTORIZED BACKTEST
# ============================================================================
def fast_backtest(params: dict, market_data: dict) -> dict:
    """
    Fast vectorized backtest given parameter set.
    Returns dict with: return, max_dd, sharpe, win_rate
    """
    conf_thresh = params['confidence_threshold']
    long_thresh = params['threshold_long']
    short_thresh = params['threshold_short']
    stop_loss = params['stop_loss_pct']

    bull_probs = market_data['bull_probs']
    returns = market_data['returns']
    start_idx = market_data['start_idx']
    n = len(returns)

    # Generate positions
    positions = np.zeros(n)
    current_pos = 0
    cumulative_loss = 0.0

    for i in range(start_idx, n):
        prob = bull_probs[i]
        conf = max(prob, 1 - prob)  # Confidence = max(P(bull), P(bear))

        target_pos = current_pos  # Default: hold

        if conf >= conf_thresh:
            if prob >= long_thresh:
                target_pos = 1   # Long
            elif prob <= short_thresh:
                target_pos = -1  # Short
            else:
                target_pos = 0   # Cash (uncertain zone)
        else:
            target_pos = 0  # Low confidence → cash

        current_pos = target_pos
        positions[i] = current_pos

        # Stop-loss check
        if current_pos != 0:
            day_pnl = current_pos * returns[i]
            cumulative_loss += day_pnl
            if cumulative_loss < -stop_loss:
                current_pos = 0
                positions[i] = 0
                cumulative_loss = 0
        else:
            cumulative_loss = 0

    # Calculate strategy returns (shift position by 1 for signal lag)
    strategy_returns = np.zeros(n)
    for i in range(start_idx + 1, n):
        strategy_returns[i] = positions[i - 1] * returns[i]

    # Transaction costs
    pos_changes = np.abs(np.diff(positions))
    pos_changes = np.concatenate([[0], pos_changes])
    txn_costs = pos_changes * 0.001  # 0.1% per trade
    strategy_returns -= txn_costs

    # Slice to trading period only
    sr = strategy_returns[start_idx:]
    br = returns[start_idx:]

    if len(sr) == 0 or np.all(sr == 0):
        return {'return': 0, 'max_dd': -1, 'sharpe': 0, 'win_rate': 0, 'trades': 0}

    # 1. Total Return
    cum_ret = np.prod(1 + sr) - 1
    total_return = cum_ret * 100  # percentage

    # 2. Max Drawdown
    equity_curve = INITIAL_CAPITAL * np.cumprod(1 + sr)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown) * 100  # negative percentage

    # 3. Sharpe Ratio (annualized)
    if np.std(sr) > 0:
        sharpe = np.mean(sr) / np.std(sr) * np.sqrt(252)
    else:
        sharpe = 0

    # 4. Win Rate (of days with non-zero returns on strategy)
    active_days = sr[sr != 0]
    if len(active_days) > 0:
        win_rate = np.mean(active_days > 0) * 100
    else:
        win_rate = 0

    # Trade count
    trades = int(np.sum(pos_changes[start_idx:] > 0))

    return {
        'return': total_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades': trades,
    }


# ============================================================================
# PYMOO PROBLEM DEFINITION
# ============================================================================
class TradingProblem(ElementwiseProblem):
    """
    4 Decision Variables × 4 Objectives

    Variables:
      x[0] = confidence_threshold  [0.50, 0.70]
      x[1] = threshold_long        [0.52, 0.70]
      x[2] = threshold_short       [0.30, 0.48]
      x[3] = stop_loss_pct         [0.02, 0.15]

    Objectives (all minimized):
      f[0] = -return         (maximize return)
      f[1] = |max_dd|        (minimize drawdown)
      f[2] = -sharpe         (maximize sharpe)
      f[3] = -win_rate       (maximize win rate)

    Constraints:
      g[0] = threshold_short - threshold_long + 0.04 <= 0
             (ensure long > short by at least 0.04)
    """

    def __init__(self, market_data: dict):
        super().__init__(
            n_var=4,
            n_obj=4,
            n_ieq_constr=1,
            xl=np.array([0.50, 0.52, 0.30, 0.02]),  # lower bounds
            xu=np.array([0.70, 0.70, 0.48, 0.15]),  # upper bounds
        )
        self.market_data = market_data

    def _evaluate(self, x, out, *args, **kwargs):
        params = {
            'confidence_threshold': x[0],
            'threshold_long': x[1],
            'threshold_short': x[2],
            'stop_loss_pct': x[3],
        }

        result = fast_backtest(params, self.market_data)

        # Objectives (pymoo minimizes by default)
        out["F"] = np.array([
            -result['return'],       # Maximize return → minimize negative
            abs(result['max_dd']),    # Minimize drawdown (already negative, take abs)
            -result['sharpe'],       # Maximize sharpe → minimize negative
            -result['win_rate'],     # Maximize win rate → minimize negative
        ])

        # Constraint: threshold_long > threshold_short + 0.04
        out["G"] = np.array([
            x[2] - x[1] + 0.04  # short - long + 0.04 <= 0
        ])


# ============================================================================
# OPTIMIZATION
# ============================================================================
def run_optimization(market: str, pop_size: int = 40, n_gen: int = 50,
                     seed: int = 42, verbose: bool = True):
    """Run NSGA-II optimization for a single market."""

    print(f"\n{'='*70}")
    print(f"🎯 MOO OPTIMIZATION: {market}")
    print(f"   Algorithm: NSGA-II | Pop: {pop_size} | Generations: {n_gen}")
    print(f"{'='*70}")

    # Prepare data
    data = prepare_market_data(market)
    if data is None:
        print(f"❌ Failed to prepare data for {market}")
        return None

    # Define problem
    problem = TradingProblem(data)

    # Configure NSGA-II
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # Run optimization
    print(f"\n🔄 Running NSGA-II ({n_gen} generations)...")
    res = pymoo_minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=verbose,
        save_history=False,
    )

    print(f"\n✅ Optimization complete!")
    print(f"   Pareto front size: {len(res.F)} solutions")

    return res


# ============================================================================
# SOLUTION SELECTION
# ============================================================================
def select_best_solutions(res, preference='balanced'):
    """
    Select best solution from Pareto front.

    Preferences:
      'balanced'   - Equal weight on all objectives (pseudo-weight)
      'max_return' - Prioritize return
      'min_risk'   - Prioritize low drawdown + high Sharpe
      'high_winrate' - Prioritize win rate
    """
    F = res.F
    X = res.X

    if preference == 'max_return':
        weights = np.array([0.6, 0.1, 0.2, 0.1])
    elif preference == 'min_risk':
        weights = np.array([0.1, 0.4, 0.4, 0.1])
    elif preference == 'high_winrate':
        weights = np.array([0.2, 0.1, 0.2, 0.5])
    else:  # balanced
        weights = np.array([0.25, 0.25, 0.25, 0.25])

    # ASF (Achievement Scalarizing Function)
    decomp = ASF()

    # Normalize F to [0, 1] range
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    F_range = F_max - F_min
    F_range[F_range == 0] = 1  # Prevent division by zero
    F_norm = (F - F_min) / F_range

    idx = decomp.do(F_norm, 1 / weights).argmin()

    best_x = X[idx]
    best_f = F[idx]

    params = {
        'confidence_threshold': round(best_x[0], 4),
        'threshold_long': round(best_x[1], 4),
        'threshold_short': round(best_x[2], 4),
        'stop_loss_pct': round(best_x[3], 4),
    }

    metrics = {
        'return': round(-best_f[0], 2),
        'max_dd': round(-best_f[1], 2),
        'sharpe': round(-best_f[2], 2),
        'win_rate': round(-best_f[3], 2),
    }

    return params, metrics, idx


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_pareto_front(res, market: str, best_idx: int = None, save_dir: str = None):
    """Plot Pareto front visualizations."""
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'moo_results')
    os.makedirs(save_dir, exist_ok=True)

    F = res.F
    # Convert to positive values for display
    returns = -F[:, 0]
    drawdowns = F[:, 1]  # Already positive (abs was taken)
    sharpes = -F[:, 2]
    win_rates = -F[:, 3]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'🎯 Pareto Front — {market}\n'
                 f'NSGA-II | {len(F)} solutions',
                 fontsize=16, fontweight='bold')

    # 1. Return vs Drawdown
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(returns, drawdowns, c=sharpes, cmap='RdYlGn',
                           s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
    if best_idx is not None:
        ax1.scatter([returns[best_idx]], [drawdowns[best_idx]],
                    c='red', s=200, marker='*', zorder=10, label='Selected')
    ax1.set_xlabel('Return (%)', fontsize=12)
    ax1.set_ylabel('Max Drawdown (%)', fontsize=12)
    ax1.set_title('Return vs Max Drawdown', fontsize=13, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Sharpe Ratio')
    ax1.grid(True, alpha=0.3)
    if best_idx is not None:
        ax1.legend(fontsize=10)

    # 2. Sharpe vs Win Rate
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(sharpes, win_rates, c=returns, cmap='viridis',
                           s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
    if best_idx is not None:
        ax2.scatter([sharpes[best_idx]], [win_rates[best_idx]],
                    c='red', s=200, marker='*', zorder=10, label='Selected')
    ax2.set_xlabel('Sharpe Ratio', fontsize=12)
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_title('Sharpe vs Win Rate', fontsize=13, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Return (%)')
    ax2.grid(True, alpha=0.3)
    if best_idx is not None:
        ax2.legend(fontsize=10)

    # 3. Return vs Sharpe
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(returns, sharpes, c=win_rates, cmap='coolwarm',
                           s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
    if best_idx is not None:
        ax3.scatter([returns[best_idx]], [sharpes[best_idx]],
                    c='red', s=200, marker='*', zorder=10, label='Selected')
    ax3.set_xlabel('Return (%)', fontsize=12)
    ax3.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3.set_title('Return vs Sharpe', fontsize=13, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, label='Win Rate (%)')
    ax3.grid(True, alpha=0.3)
    if best_idx is not None:
        ax3.legend(fontsize=10)

    # 4. Parameter Distribution (Parallel Coordinates)
    ax4 = axes[1, 1]
    X = res.X
    param_names = ['Confidence', 'Long Thresh', 'Short Thresh', 'Stop Loss']

    # Normalize to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range

    for i in range(len(X)):
        color = plt.cm.viridis(returns[i] / max(returns.max(), 1))
        alpha = 0.3
        if best_idx is not None and i == best_idx:
            color = 'red'
            alpha = 1.0
        ax4.plot(range(4), X_norm[i], color=color, alpha=alpha, linewidth=1)

    ax4.set_xticks(range(4))
    ax4.set_xticklabels(param_names, fontsize=10)
    ax4.set_ylabel('Normalized Value', fontsize=12)
    ax4.set_title('Parameter Distribution (Parallel Coords)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, f'pareto_front_{market}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Pareto front chart saved: {filepath}")
    return filepath


# ============================================================================
# COMPARISON: OPTIMIZED vs ORIGINAL
# ============================================================================
def compare_params(market: str, optimized_params: dict, original_params: dict = None):
    """Run backtest with both parameter sets and compare."""
    if original_params is None:
        original_params = ORIGINAL_PARAMS

    data = prepare_market_data(market)
    if data is None:
        return None

    result_original = fast_backtest(original_params, data)
    result_optimized = fast_backtest(optimized_params, data)

    comparison = {
        'Metric': ['Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Trades'],
        'Original': [
            f"{result_original['return']:.2f}",
            f"{result_original['max_dd']:.2f}",
            f"{result_original['sharpe']:.2f}",
            f"{result_original['win_rate']:.1f}",
            result_original['trades'],
        ],
        'Optimized': [
            f"{result_optimized['return']:.2f}",
            f"{result_optimized['max_dd']:.2f}",
            f"{result_optimized['sharpe']:.2f}",
            f"{result_optimized['win_rate']:.1f}",
            result_optimized['trades'],
        ],
        'Change': [
            f"{result_optimized['return'] - result_original['return']:+.2f}",
            f"{result_optimized['max_dd'] - result_original['max_dd']:+.2f}",
            f"{result_optimized['sharpe'] - result_original['sharpe']:+.2f}",
            f"{result_optimized['win_rate'] - result_original['win_rate']:+.1f}",
            f"{result_optimized['trades'] - result_original['trades']:+d}",
        ]
    }

    return pd.DataFrame(comparison)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='MOO Trading Strategy Optimizer')
    parser.add_argument('--market', type=str, default=None,
                        help='Single market to optimize (e.g., US, UK, Thai, Gold, BTC)')
    parser.add_argument('--pop_size', type=int, default=40,
                        help='Population size for NSGA-II (default: 40)')
    parser.add_argument('--generations', type=int, default=50,
                        help='Number of generations (default: 50)')
    parser.add_argument('--preference', type=str, default='balanced',
                        choices=['balanced', 'max_return', 'min_risk', 'high_winrate'],
                        help='Solution selection preference')
    parser.add_argument('--compare', action='store_true',
                        help='Compare optimized vs original parameters')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    markets = [args.market] if args.market else list(TICKERS.keys())

    save_dir = os.path.join(os.path.dirname(__file__), 'moo_results')
    os.makedirs(save_dir, exist_ok=True)

    all_results = []

    print("=" * 70)
    print("🎯 MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)")
    print(f"   Markets: {', '.join(markets)}")
    print(f"   Pop: {args.pop_size} | Gen: {args.generations} | Preference: {args.preference}")
    print("=" * 70)

    for market in markets:
        # Run optimization
        res = run_optimization(
            market,
            pop_size=args.pop_size,
            n_gen=args.generations,
            seed=args.seed,
            verbose=False,
        )

        if res is None:
            continue

        # Select best solution
        best_params, best_metrics, best_idx = select_best_solutions(res, args.preference)

        print(f"\n  🏆 Best Solution ({args.preference}):")
        print(f"     Parameters:")
        for k, v in best_params.items():
            orig_v = ORIGINAL_PARAMS.get(k, '?')
            print(f"       {k:25s} = {v:.4f}  (was {orig_v})")
        print(f"     Metrics:")
        for k, v in best_metrics.items():
            print(f"       {k:15s} = {v:.2f}")

        # Plot Pareto front
        chart_path = plot_pareto_front(res, market, best_idx, save_dir)

        # Compare if requested
        if args.compare:
            print(f"\n  📊 Comparison — {market}:")
            comp_df = compare_params(market, best_params)
            if comp_df is not None:
                print(comp_df.to_string(index=False))

        # Collect results
        all_results.append({
            'Market': market,
            'Preference': args.preference,
            **best_params,
            **{f'metric_{k}': v for k, v in best_metrics.items()},
            'pareto_size': len(res.F),
        })

    # Save summary
    if all_results:
        df_results = pd.DataFrame(all_results)
        csv_path = os.path.join(save_dir, 'moo_optimized_params.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"\n{'='*70}")
        print(f"📋 SUMMARY")
        print(f"{'='*70}")
        print(df_results.to_string(index=False))
        print(f"\n📁 Results saved to: {save_dir}/")
        print(f"   - moo_optimized_params.csv")
        for market in markets:
            print(f"   - pareto_front_{market}.png")

        # Generate config snippet
        print(f"\n{'='*70}")
        print("📝 CONFIG.PY SNIPPET (copy-paste ready):")
        print(f"{'='*70}")
        print("\n# MOO-Optimized Parameters (per market)")
        print("OPTIMIZED_PARAMS = {")
        for r in all_results:
            print(f"    '{r['Market']}': {{"
                  f"'confidence': {r['confidence_threshold']:.4f}, "
                  f"'long': {r['threshold_long']:.4f}, "
                  f"'short': {r['threshold_short']:.4f}, "
                  f"'stop_loss': {r['stop_loss_pct']:.4f}}},")
        print("}")


if __name__ == "__main__":
    main()
