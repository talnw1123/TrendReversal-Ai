import pandas as pd
import os

def analyze_periods():
    markets = ['US', 'UK', 'Thai', 'Gold', 'BTC']
    
    print("Analyzing trend periods (Price vs SMA200)...")
    
    for market in markets:
        filename = f"{market}_full_history.csv"
        if not os.path.exists(filename):
            print(f"Skipping {market}: {filename} not found.")
            continue
            
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        # Handle potential column naming issues from fetch
        # We expect one column or we take the first one
        if len(df.columns) > 1:
            # Try to find a 'Close' column or similar
            col = [c for c in df.columns if 'Close' in c]
            if col:
                price = df[col[0]]
            else:
                price = df.iloc[:, 0]
        else:
            price = df.iloc[:, 0]
            
        # Calculate SMA200
        sma = price.rolling(window=200).mean()
        
        # Identify trends
        is_uptrend = price > sma
        is_downtrend = price <= sma
        
        # Find continuous periods
        # We'll create a dataframe to track changes
        trend_df = pd.DataFrame({'uptrend': is_uptrend, 'downtrend': is_downtrend})
        
        # Group by consecutive values
        # False->True or True->False changes
        trend_df['group_up'] = (trend_df['uptrend'] != trend_df['uptrend'].shift()).cumsum()
        trend_df['group_down'] = (trend_df['downtrend'] != trend_df['downtrend'].shift()).cumsum()
        
        print(f"\n--- {market} ---")
        
        # Analyze Uptrends
        print("Longest Uptrends:")
        up_groups = trend_df[trend_df['uptrend']].groupby('group_up')
        for _, group in sorted(up_groups, key=lambda x: len(x[1]), reverse=True)[:3]:
            start = group.index.min().date()
            end = group.index.max().date()
            duration = (group.index.max() - group.index.min()).days / 365.25
            print(f"  {start} to {end} ({duration:.1f} years)")
            
        # Analyze Downtrends
        print("Longest Downtrends:")
        down_groups = trend_df[trend_df['downtrend']].groupby('group_down')
        for _, group in sorted(down_groups, key=lambda x: len(x[1]), reverse=True)[:3]:
            start = group.index.min().date()
            end = group.index.max().date()
            duration = (group.index.max() - group.index.min()).days / 365.25
            print(f"  {start} to {end} ({duration:.1f} years)")

if __name__ == "__main__":
    analyze_periods()
