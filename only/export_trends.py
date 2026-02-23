import pandas as pd
import os

def export_trends():
    print("Loading data...")
    try:
        df = pd.read_csv("combined_market_data.csv", index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("Error: combined_market_data.csv not found. Please run fetch_data.py first.")
        return

    # Create output directory
    output_dir = "trend_data"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting trend data to '{output_dir}/'...")

    markets = ['US', 'UK', 'China', 'Thai', 'BTC']

    for market in markets:
        col_close = f"{market}_Close"
        col_sma = f"{market}_SMA200"
        
        # Calculate SMA if not already present (though plot_trends calculates it, we need it here)
        # Note: combined_market_data.csv might only have OHLCV data, so we recalculate.
        df[col_sma] = df[col_close].rolling(window=200).mean()

        # Drop NaN values for this market's SMA to ensure clean trend data
        market_df = df.dropna(subset=[col_sma]).copy()

        if market_df.empty:
            print(f"Warning: No data for {market} after SMA calculation.")
            continue

        # Identify Trends
        uptrend_mask = market_df[col_close] > market_df[col_sma]
        downtrend_mask = market_df[col_close] <= market_df[col_sma]

        # Filter Data
        uptrend_df = market_df[uptrend_mask]
        downtrend_df = market_df[downtrend_mask]

        # Save to CSV
        uptrend_file = os.path.join(output_dir, f"{market}_uptrend.csv")
        downtrend_file = os.path.join(output_dir, f"{market}_downtrend.csv")

        uptrend_df.to_csv(uptrend_file)
        downtrend_df.to_csv(downtrend_file)

        print(f"  {market}: Exported {len(uptrend_df)} uptrend rows and {len(downtrend_df)} downtrend rows.")

    print("Export complete.")

if __name__ == "__main__":
    export_trends()
