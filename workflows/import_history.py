import sqlite3
import json
import os
import sys

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plot')
db_path = os.path.join(script_dir, 'trading_database.sqlite')

markets = ['US', 'UK', 'Thai', 'Gold', 'BTC']

def import_history_to_db():
    print("=" * 60)
    print(" 📥 IMPORTING HISTORICAL SIGNALS TO SQLITE")
    print("=" * 60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_inserted = 0
    
    for market in markets:
        json_file = os.path.join(plots_dir, f"{market}_history.json")
        if not os.path.exists(json_file):
            print(f"⚠️  Missing {json_file}. Did you run 'python workflows/trading_system.py --backtest'?")
            continue
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        table_name = f"signals_history_{market}"
        
        # Ensure table exists with correct schema
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
        
        # Clear existing data for this market to prevent duplicates
        cursor.execute(f'DELETE FROM "{table_name}"')
        
        print(f"📡 Importing {len(data)} trading days for {market}...")
        
        for row in data:
            cursor.execute(f'''
                INSERT OR IGNORE INTO "{table_name}" 
                (date, market, price, trend_regime, ml_up_prob, ml_down_prob, signal_action, position, equity_curve, bnh_curve)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['date'], 
                row.get('market', market),
                row.get('price', 0),
                row.get('trend_regime', ''),
                row.get('ml_up_prob', 0),
                row.get('ml_down_prob', 0),
                row.get('signal_action', ''),
                row.get('position', 0),
                row.get('equity_curve', 0),
                row.get('bnh_curve', 0),
            ))
            total_inserted += 1
            
    conn.commit()
    conn.close()
    
    print("=" * 60)
    print(f"✅ Successfully imported {total_inserted} historical records into Database!")
    print("=" * 60)

if __name__ == "__main__":
    import_history_to_db()
