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
    print(" 📥 IMPORTING 2-YEAR HISTORICAL SIGNALS TO SQLITE")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}. Please run llm_agent.py first to init.")
        sys.exit(1)
        
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
            
        print(f"📡 Importing {len(data)} trading days for {market}...")
        
        # Insert ignores duplicate dates for the same market via UNIQUE constraint (assume we defined it)
        # Wait, the current schema might not have a UNIQUE constraint. Let's delete existing first to prevent duplicates
        cursor.execute("DELETE FROM signals_history WHERE market = ?", (market,))
        
        for row in data:
            cursor.execute('''
                INSERT INTO signals_history (date, market, trend, signal_text, price)
                VALUES (?, ?, ?, ?, ?)
            ''', (row['date'], row['market'], row['trend'], row['signal_text'], row['price']))
            total_inserted += 1
            
    conn.commit()
    conn.close()
    
    print("=" * 60)
    print(f"✅ Successfully imported {total_inserted} historical records into Database!")
    print("=" * 60)

if __name__ == "__main__":
    import_history_to_db()
