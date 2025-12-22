import pandas as pd
import os
import glob

MARKETS = ['US', 'UK', 'Thai', 'Gold', 'BTC']

def load_full_market_data(market):
    """Load full price history for a market (train + val + test)."""
    # Simply load and concat like in smart_backtest.py
    # We don't need features, just dates
    
    all_data = []
    # Assuming script is run from root or model dir, try to find the path
    # smart_backtest.py uses: os.path.join(script_dir, "../trend_data_manual/split", split)
    # We will assume we are running from project root
    
    base_search_path = "trend_data_manual/split"
    
    for split in ['train', 'val', 'test']:
        for trend in ['uptrend', 'downtrend']:
            pattern = os.path.join(base_search_path, split, "**", f"{market}_{trend}_labeled.csv")
            files = glob.glob(pattern, recursive=True)
            for f in files:
                try:
                    df = pd.read_csv(f, index_col=0, parse_dates=True)
                    all_data.append(df)
                except Exception as e:
                    pass
    
    if not all_data:
        return None
    
    # Combine and sort
    df_combined = pd.concat(all_data)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')].sort_index()
    
    return df_combined

def check_dates():
    print(f"{'Market':<10} {'Start Date':<25} {'End Date':<25}")
    print("-" * 60)
    
    for market in MARKETS:
        df = load_full_market_data(market)
        if df is None:
            print(f"{market:<10} {'No Data':<50}")
            continue
            
        total_len = len(df)
        test_start_idx = int(total_len * 0.85)
        
        test_df = df.iloc[test_start_idx:]
        
        if test_df.empty:
             print(f"{market:<10} {'Empty Test Set':<50}")
             continue
             
        start_date = test_df.index[0].strftime('%Y-%m-%d')
        end_date = test_df.index[-1].strftime('%Y-%m-%d')
        
        print(f"{market:<10} {start_date:<25} {end_date:<25}")

if __name__ == "__main__":
    check_dates()
