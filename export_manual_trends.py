import pandas as pd
import os

def export_manual_trends():
    print("Loading full history data...")
    
    # Define the manual periods (Start Date, End Date)
    # Targeting ~6-7 years where possible
    periods = {
        'US': {
            'uptrend': ('2010-01-01', '2017-12-31'), # Post-GFC Bull Run (8 years)
            'downtrend': ('2000-01-01', '2009-03-09') # Dotcom + GFC (The Lost Decade) (~9 years)
        },
        'UK': {
            'uptrend': ('2009-03-01', '2015-05-01'), # Post-GFC Recovery (~6 years)
            'downtrend': ('2000-01-01', '2009-02-28') # Volatile/Down decade (~9 years)
        },
        'Thai': {
            'uptrend': ('2009-01-01', '2015-01-01'), # Post-GFC Boom (6 years)
            'downtrend': ('1994-01-01', '2001-01-01') # Tom Yum Kung Crisis Era (7 years)
        },
        'Gold': {
            'uptrend': ('2005-01-01', '2011-08-31'), # The Great Gold Bull Run (~6.5 years)
            'downtrend': ('2011-09-01', '2018-08-01') # The Bear Market/Correction (~7 years)
        },
        'BTC': {
            'uptrend': ('2013-01-01', '2017-12-15'), # Early Adopters + 2017 Boom (Ends before crash)
            'downtrend': ('2018-01-01', '2019-12-31') # The 2018 Crash + Crypto Winter (Distinct period)
        }
    }

    output_dir = "trend_data_manual"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting manual trend data to '{output_dir}/'...")

    for market, trends in periods.items():
        # Load the specific full history file
        filename = f"{market}_full_history.csv"
        if not os.path.exists(filename):
            print(f"Error: {filename} not found. Skipping {market}.")
            continue
            
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        # We want all columns (OHLCV), so we don't filter for just 'Close' anymore.
        # However, we might want to ensure the index name is 'Date'
        df.index.name = 'Date'

        for trend_type, (start, end) in trends.items():
            # Slice data
            mask = (df.index >= start) & (df.index <= end)
            trend_df = df.loc[mask]
            
            if trend_df.empty:
                print(f"  Warning: No data found for {market} {trend_type} ({start} to {end})")
                continue
                
            output_file = os.path.join(output_dir, f"{market}_{trend_type}.csv")
            trend_df.to_csv(output_file)
            
            years = (trend_df.index.max() - trend_df.index.min()).days / 365.25
            print(f"  {market} {trend_type}: {start} to {end} ({years:.1f} years, {len(trend_df)} rows)")

    print("Manual export complete.")

if __name__ == "__main__":
    export_manual_trends()
