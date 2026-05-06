"""
Trading Chart Visualization
- Shows price chart with trend detection
- Marks trading signals on chart
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../model'))

from config import TICKERS, BEST_MODELS, TREND_METHOD
from model.regime_detection import RegimeDetector


def create_trading_chart(market: str, use_max_history: bool = True):
    """Create trading chart for a specific market"""
    
    ticker = TICKERS.get(market)
    if not ticker:
        print(f"Unknown market: {market}")
        return
    
    print(f"Creating chart for {market}...")
    
    # Fetch ALL historical data
    df = yf.download(ticker, period="max", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        print(f"No data for {market}")
        return
    
    # Detect trend
    if TREND_METHOD == 'adx_supertrend':
        is_uptrend = RegimeDetector.detect_adx_supertrend(df)
    elif TREND_METHOD == 'gmm':
        is_uptrend = RegimeDetector.detect_gmm(df)
    else:
        sma200 = df['Close'].rolling(200).mean()
        is_uptrend = (df['Close'] > sma200).astype(int).values
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{market} Trading Chart ({ticker})', fontsize=16, fontweight='bold')
    
    # Main price chart
    ax1 = axes[0]
    
    # Plot price
    ax1.plot(df.index, df['Close'], label='Close Price', color='#2196F3', linewidth=2)
    
    # Plot moving averages
    ma20 = df['Close'].rolling(20).mean()
    ma60 = df['Close'].rolling(60).mean()
    ax1.plot(df.index, ma20, label='MA20', color='#FF9800', linewidth=1, alpha=0.8)
    ax1.plot(df.index, ma60, label='MA60', color='#9C27B0', linewidth=1, alpha=0.8)
    
    # Color background based on trend
    for i in range(1, len(df)):
        if is_uptrend[i] == 1:
            ax1.axvspan(df.index[i-1], df.index[i], alpha=0.1, color='green')
        else:
            ax1.axvspan(df.index[i-1], df.index[i], alpha=0.1, color='red')
    
    # Mark trend changes
    trend_changes = []
    for i in range(1, len(is_uptrend)):
        if is_uptrend[i] != is_uptrend[i-1]:
            trend_changes.append((df.index[i], is_uptrend[i]))
    
    for date, trend in trend_changes:
        color = 'green' if trend == 1 else 'red'
        marker = '^' if trend == 1 else 'v'
        ax1.scatter([date], [df.loc[date, 'Close']], color=color, marker=marker, s=150, zorder=5)
    
    # Current trend and signal
    current_trend = 'UPTREND' if is_uptrend[-1] == 1 else 'DOWNTREND'
    best_model = BEST_MODELS.get(market, {}).get('uptrend' if is_uptrend[-1] == 1 else 'downtrend', 'N/A')
    
    # Add annotations
    ax1.text(0.02, 0.98, f'Current Trend: {current_trend}', 
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             color='green' if current_trend == 'UPTREND' else 'red')
    
    ax1.text(0.02, 0.90, f'Best Model: {best_model}', 
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Use YearLocator for long-term data
    years = (df.index[-1] - df.index[0]).days / 365
    if years > 10:
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif years > 5:
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Volume chart
    ax2 = axes[1]
    colors = ['green' if is_uptrend[i] == 1 else 'red' for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=1.0)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Same x-axis format as price chart
    if years > 10:
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif years > 5:
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(script_dir, f'chart_{market}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {chart_path}")
    
    plt.close()
    return chart_path


def create_all_charts():
    """Create charts for all markets"""
    print("=" * 60)
    print("CREATING TRADING CHARTS")
    print("=" * 60)
    
    chart_paths = []
    for market in TICKERS.keys():
        path = create_trading_chart(market)
        if path:
            chart_paths.append(path)
    
    print("\n" + "=" * 60)
    print("CHARTS CREATED:")
    print("=" * 60)
    for path in chart_paths:
        print(f"  - {path}")
    
    return chart_paths


if __name__ == "__main__":
    create_all_charts()
