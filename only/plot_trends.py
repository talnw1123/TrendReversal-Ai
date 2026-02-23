import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_trends():
    print("Loading data...")
    df = pd.read_csv("combined_market_data.csv", index_col='Date', parse_dates=True)
    
    # Calculate SMA200 for all markets
    markets = [
        {'name': 'US', 'label': 'S&P 500 (US)', 'color': 'black'},
        {'name': 'UK', 'label': 'FTSE 100 (UK)', 'color': 'blue'},
        {'name': 'China', 'label': 'SSE Composite (China)', 'color': 'red'},
        {'name': 'Thai', 'label': 'SET Index (Thailand)', 'color': 'orange'},
        {'name': 'BTC', 'label': 'Bitcoin (BTC)', 'color': 'purple'}
    ]
    
    for m in markets:
        col_name = f"{m['name']}_Close"
        sma_name = f"{m['name']}_SMA200"
        df[sma_name] = df[col_name].rolling(window=200).mean()

    # Use US SMA for reference trimming
    df['US_SMA200'] = df['US_Close'].rolling(window=200).mean()
    plot_df = df.dropna(subset=['US_SMA200'])
    
    print(f"Plotting data from {plot_df.index.min()} to {plot_df.index.max()}...")
    
    # Loop through markets and create a separate plot for each
    for m in markets:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        col_name = f"{m['name']}_Close"
        sma_name = f"{m['name']}_SMA200"
        
        # Define Uptrend/Downtrend masks for THIS market
        is_uptrend = plot_df[col_name] > plot_df[sma_name]
        is_downtrend = plot_df[col_name] <= plot_df[sma_name]
        
        # Plot Price
        ax.plot(plot_df.index, plot_df[col_name], label=m['label'], color=m['color'], linewidth=1.5)
        
        # Plot SMA200
        ax.plot(plot_df.index, plot_df[sma_name], label='200-day SMA', color='gray', linestyle='--', linewidth=1)
        
        # Fill areas based on THIS Market's Trend
        # Update limits before shading
        ax.relim()
        ax.autoscale_view()
        
        # Uptrend shading
        ax.fill_between(
            plot_df.index, 0, 1,
            where=is_uptrend,
            interpolate=True, color='green', alpha=0.15,
            transform=ax.get_xaxis_transform()
        )
        
        # Downtrend shading
        ax.fill_between(
            plot_df.index, 0, 1,
            where=is_downtrend,
            interpolate=True, color='red', alpha=0.15,
            transform=ax.get_xaxis_transform()
        )
        
        ax.set_title(f"{m['label']} - Trend Analysis", fontsize=16)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.5)
        ax.set_yscale('log')
        
        # Format Y-axis to show full numbers (no scientific notation)
        from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter()) # Disable minor labels
        
        # Ensure ticks at 1, 2, 5 intervals (e.g., 1000, 2000, 5000)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
        
        ax.ticklabel_format(style='plain', axis='y')
        
        # X-axis formatting
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (Log Scale)', fontsize=12)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()
        
        output_file = f"market_trend_{m['name']}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close(fig)

if __name__ == "__main__":
    plot_trends()