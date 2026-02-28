import pandas as pd
import sqlite3
import os
from datetime import datetime

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
db_path = os.path.join(script_dir, 'trading_database.sqlite')
csv_path = os.path.join(project_root, 'model', 'smart_backtest_results.csv')

def populate_performance():
    print("=" * 60)
    print(" 📊 SYNCING PERFORMANCE METRICS: CSV -> SQLITE")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV not found at {csv_path}")
        return

    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter for Overall results (not just Uptrend/Downtrend splits)
    # We want the 'Overall' row for each Market/Regime/Model combination
    overall_df = df[df['Trend'] == 'Overall'].copy()
    
    # For each market, find the model with the highest 'Strategy Return (%)'
    best_models = []
    markets = ['US', 'UK', 'Thai', 'Gold', 'BTC']
    
    for market in markets:
        market_stats = overall_df[overall_df['Market'] == market]
        if market_stats.empty:
            print(f"⚠️  No 'Overall' data found for market: {market}")
            continue
            
        # Select row with maximum Strategy Return (%)
        best_row = market_stats.loc[market_stats['Strategy Return (%)'].idxmax()]
        best_models.append(best_row)
        print(f"✅ Best for {market}: {best_row['Model']} (Return: {best_row['Strategy Return (%)']}%)")

    if not best_models:
        print("❌ No best models found to sync.")
        return

    # Connect to Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ensure table exists (schema matches web_dashboard.py expectation)
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
    
    # Insert or Replace
    for row in best_models:
        # Map CSV columns to DB columns
        # CSV: Market,Regime,Trend,Model,Strategy Return (%),B&H Return (%),Max DD (%),Final $,Long (%),Short (%),Turnover,Tie Warning
        # win_rate_pct isn't explicitly in this csv, we'll use 0 or try to calculate if it was there. 
        # Actually in typical quant csv 'Long (%)' + 'Short (%)' might be related, but win_rate is likely not stored here.
        # Let's check columns again carefully.
        
        # We'll use 0.0 for win_rate_pct as it's missing from this specific CSV structure
        win_rate = 0.0 
        
        cursor.execute('''
            INSERT OR REPLACE INTO strategy_performance 
            (market, base_return_pct, bnh_return_pct, win_rate_pct, total_trades, max_drawdown_pct, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['Market'],
            float(row['Strategy Return (%)']),
            float(row['B&H Return (%)']),
            win_rate,
            int(row['Turnover']) if not pd.isna(row['Turnover']) else 0,
            float(row['Max DD (%)']),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
    conn.commit()
    conn.close()
    print("=" * 60)
    print("✨ Database successfully updated with best model performance!")
    print("=" * 60)

if __name__ == "__main__":
    populate_performance()
