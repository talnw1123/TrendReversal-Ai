"""
populate_metrics.py
====================
Sync strategy_performance table from signals_history_* tables.

This reads the actual equity curve and BnH curve data written by trading_system.py
(which matches the reference PNGs) and computes:
  - Strategy Return %
  - Buy & Hold Return %
  - Win Rate % (from BUY→SELL trade cycles)
  - Total Trades
  - Max Drawdown %
"""
import sqlite3
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, 'trading_database.sqlite')

MARKETS = ['US', 'UK', 'Thai', 'Gold', 'BTC']


def populate_performance():
    print("=" * 60)
    print(" 📊 SYNCING PERFORMANCE METRICS: signals_history -> strategy_performance")
    print("=" * 60)

    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure table exists
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

    for market in MARKETS:
        table_name = f"signals_history_{market}"

        # Check table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            print(f"⚠️  Table '{table_name}' not found, skipping {market}")
            continue

        # Get all rows ordered by date
        cursor.execute(f'SELECT date, price, position, equity_curve, bnh_curve FROM "{table_name}" ORDER BY date ASC')
        rows = cursor.fetchall()

        if not rows:
            print(f"⚠️  No data in '{table_name}', skipping {market}")
            continue

        # ── Strategy Return & Buy & Hold Return ──
        # Use first and last equity/bnh values
        first_eq = None
        last_eq = None
        first_bnh = None
        last_bnh = None

        for row in rows:
            eq = row[3]
            bnh = row[4]
            if eq and eq > 0:
                if first_eq is None:
                    first_eq = eq
                last_eq = eq
            if bnh and bnh > 0:
                if first_bnh is None:
                    first_bnh = bnh
                last_bnh = bnh

        if first_eq and last_eq and first_eq > 0:
            strategy_return = (last_eq - first_eq) / first_eq * 100
        else:
            strategy_return = 0.0

        if first_bnh and last_bnh and first_bnh > 0:
            bnh_return = (last_bnh - first_bnh) / first_bnh * 100
        else:
            bnh_return = 0.0

        # ── Max Drawdown ──
        equity_values = [r[3] for r in rows if r[3] and r[3] > 0]
        max_dd = 0.0
        if equity_values:
            peak = equity_values[0]
            for val in equity_values:
                if val > peak:
                    peak = val
                dd = (val - peak) / peak * 100
                if dd < max_dd:
                    max_dd = dd

        # ── Win Rate (from position transitions) ──
        wins = 0
        losses = 0
        buy_price = None
        prev_pos = 0.0
        total_trades = 0

        for row in rows:
            price = float(row[1]) if row[1] else 0.0
            pos = float(row[2]) if row[2] else 0.0

            if pos > 0 and prev_pos <= 0 and price > 0:
                # Entered BUY position
                buy_price = price
                total_trades += 1
            elif pos <= 0 and prev_pos > 0 and buy_price is not None and price > 0:
                # Exited position (SELL)
                if price > buy_price:
                    wins += 1
                else:
                    losses += 1
                buy_price = None
            prev_pos = pos

        completed_trades = wins + losses
        win_rate = round((wins / completed_trades) * 100, 1) if completed_trades > 0 else 0.0

        # ── Insert into strategy_performance ──
        cursor.execute('''
            INSERT OR REPLACE INTO strategy_performance 
            (market, base_return_pct, bnh_return_pct, win_rate_pct, total_trades, max_drawdown_pct, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            market,
            round(strategy_return, 2),
            round(bnh_return, 2),
            win_rate,
            total_trades,
            round(max_dd, 2),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))

        print(f"✅ {market}: Return={strategy_return:+.2f}%, B&H={bnh_return:+.2f}%, "
              f"WR={win_rate}% ({wins}W/{losses}L), DD={max_dd:.2f}%, Trades={total_trades}")

    conn.commit()
    conn.close()
    print("=" * 60)
    print("✨ Database successfully updated from signals_history tables!")
    print("=" * 60)


if __name__ == "__main__":
    populate_performance()
