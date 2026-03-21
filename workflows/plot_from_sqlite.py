import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

# Configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_database.sqlite')
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot')
MARKETS = ['US', 'UK', 'Thai', 'Gold', 'BTC']

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def query_market_data(market):
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM signals_history_{market} ORDER BY date ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return None
        
        # Convert date column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    except sqlite3.Error as e:
        print(f"Database error for {market}: {e}")
        return None

def plot_market(market, df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    dates = df['date'].tolist()
    prices = df['price'].tolist()
    equity = df['equity_curve'].tolist()
    bnh = df['bnh_curve'].tolist()
    signals = df['signal_action'].tolist()
    regimes = df['trend_regime'].tolist()
    
    # Identify trade markers
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    
    for i in range(len(df)):
        sig = str(signals[i]).upper()
        if sig == 'BUY':
            buy_dates.append(dates[i])
            buy_prices.append(prices[i])
        elif sig == 'SELL':
            sell_dates.append(dates[i])
            sell_prices.append(prices[i])
            
    # 1. Price Chart with Markers
    ax1.plot(dates, prices, label='Price', color='black', alpha=0.6)
    if buy_dates:
        ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy/Long', zorder=5)
    if sell_dates:
        ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell/Exit', zorder=5)
    
    ax1.set_title(f"{market} - Price & Signals (from SQLite)")
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Regime Background
    if len(regimes) == len(dates):
        start_idx = 0
        current_val = '1' if '1' in str(regimes[0]) or 'Uptrend' in str(regimes[0]) else '0'
        
        for i in range(1, len(regimes)):
            val = '1' if '1' in str(regimes[i]) or 'Uptrend' in str(regimes[i]) else '0'
            if val != current_val:
                color = 'green' if current_val == '1' else 'red'
                ax1.axvspan(dates[start_idx], dates[i], color=color, alpha=0.1)
                ax2.axvspan(dates[start_idx], dates[i], color=color, alpha=0.1)
                start_idx = i
                current_val = val
                
        # Last segment
        color = 'green' if current_val == '1' else 'red'
        ax1.axvspan(dates[start_idx], dates[-1], color=color, alpha=0.1)
        ax2.axvspan(dates[start_idx], dates[-1], color=color, alpha=0.1)

    # 2. Equity Curve
    final_return = ((equity[-1] / 10000.0) - 1.0) * 100.0 if equity else 0.0
    bnh_return = ((bnh[-1] / 10000.0) - 1.0) * 100.0 if bnh else 0.0
    
    ax2.plot(dates, bnh, label=f'Buy & Hold ({bnh_return:.2f}%)', color='gray', linestyle='--')
    ax2.plot(dates, equity, label=f'Strategy ({final_return:.2f}%)', color='blue', linewidth=2)
    ax2.set_title(f"Equity Curve - {market}")
    ax2.set_ylabel('Equity ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Date formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    filename = os.path.join(PLOT_DIR, f"{market}_Combined_Hybrid_MOO.png")
    plt.savefig(filename)
    plt.close()
    print(f"📊 Plot saved: {filename}")

if __name__ == "__main__":
    print("=" * 60)
    print(" 📈 GENERATING PLOTS FROM SQLITE DATABASE")
    print("=" * 60)
    
    for market in MARKETS:
        print(f"Processing {market}...")
        df = query_market_data(market)
        if df is not None:
            plot_market(market, df)
        else:
            print(f"  ⚠️ No data found for {market}")
    
    print("✅ Done!")
