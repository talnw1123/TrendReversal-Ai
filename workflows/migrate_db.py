import sqlite3
import os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_database.sqlite')
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()

for table in tables:
    table_name = table[0]
    if table_name.startswith('signals_history_'):
        # Check if news_summary column exists
        c.execute(f"PRAGMA table_info('{table_name}')")
        columns = [col[1] for col in c.fetchall()]
        
        if 'news_summary' not in columns:
            print(f"Adding news_summary column to {table_name}...")
            c.execute(f'ALTER TABLE "{table_name}" ADD COLUMN news_summary TEXT')
        else:
            print(f"Table {table_name} already has news_summary column.")

conn.commit()
conn.close()
print("Migration completed.")
