import yfinance as yf
import pandas as pd

def fetch_and_align_data():
    # Define tickers
    tickers = {
        'US': '^GSPC',      # S&P 500
        'UK': '^FTSE',      # FTSE 100
        'Thai': '^SET.BK',  # SET Index
        'Gold': 'GC=F',     # Gold Futures
        'BTC': 'BTC-USD'    # Bitcoin
    }

    print("Fetching data...")
    data_frames = {}
    
    for name, ticker in tickers.items():
        print(f"Downloading {name} ({ticker})...")
        # Download max history
        try:
            df = yf.download(ticker, period="max", progress=False)
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            continue
        
        # Ensure we have a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Handle MultiIndex columns if present (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
             # If MultiIndex, usually level 0 is Price Type, level 1 is Ticker
             # We want to keep the Price Type (Open, Close, etc.)
             try:
                df.columns = df.columns.get_level_values(0)
             except:
                pass

        # Save full history with all columns for manual analysis later
        full_history_file = f"{name}_full_history.csv"
        df.to_csv(full_history_file)
        print(f"  Saved full history to {full_history_file} ({len(df)} rows)")

        # Prepare for combined file (Close price only)
        if 'Close' in df.columns:
            df_close = df[['Close']].copy()
            df_close.columns = [f"{name}_Close"]
            data_frames[name] = df_close
        else:
            print(f"  Warning: 'Close' column not found for {name}")

    print("Aligning data (finding common dates)...")
    if 'US' not in data_frames:
        print("Error: US data missing, cannot align.")
        return

    # Start with the first dataframe
    combined_df = data_frames['US']
    
    # Inner join with the rest
    for name in ['UK', 'Thai', 'Gold', 'BTC']:
        if name in data_frames:
            combined_df = combined_df.join(data_frames[name], how='inner')
        else:
            print(f"Warning: {name} data missing, skipping join.")

    print(f"Data aligned. Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Total rows: {len(combined_df)}")
    
    output_file = "combined_market_data.csv"
    combined_df.to_csv(output_file)
    print(f"Saved to {output_file}")
    
    print("\nFirst 5 rows:")
    print(combined_df.head())

if __name__ == "__main__":
    fetch_and_align_data()
