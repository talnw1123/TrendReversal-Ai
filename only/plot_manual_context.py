import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def plot_manual_context():
    markets = [
        {'name': 'US', 'label': 'S&P 500 (US)'},
        {'name': 'UK', 'label': 'FTSE 100 (UK)'},
        {'name': 'Thai', 'label': 'SET Index (Thailand)'},
        {'name': 'Gold', 'label': 'Gold (GC=F)'},
        {'name': 'BTC', 'label': 'Bitcoin (BTC)'}
    ]
    
    manual_dir = "trend_data_manual"
    
    print("Generating context plots...")
    
    for m in markets:
        market_name = m['name']
        full_history_file = f"{market_name}_full_history.csv"
        
        if not os.path.exists(full_history_file):
            print(f"Skipping {market_name}: {full_history_file} not found.")
            continue
            
        # Load full history
        df_full = pd.read_csv(full_history_file, index_col=0, parse_dates=True)
        
        # Handle column names (find Close)
        col = [c for c in df_full.columns if 'Close' in c]
        if col:
            price_full = df_full[col[0]]
        else:
            price_full = df_full.iloc[:, 0]
            
        # Load Manual Trends
        uptrend_file = os.path.join(manual_dir, f"{market_name}_uptrend.csv")
        downtrend_file = os.path.join(manual_dir, f"{market_name}_downtrend.csv")
        
        uptrend_range = None
        downtrend_range = None
        
        if os.path.exists(uptrend_file):
            df_up = pd.read_csv(uptrend_file, index_col=0, parse_dates=True)
            if not df_up.empty:
                uptrend_range = (df_up.index.min(), df_up.index.max())
                
        if os.path.exists(downtrend_file):
            df_down = pd.read_csv(downtrend_file, index_col=0, parse_dates=True)
            if not df_down.empty:
                downtrend_range = (df_down.index.min(), df_down.index.max())
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Full History
        ax.plot(price_full.index, price_full, color='lightgray', linewidth=1, label='Full History')
        
        # Highlight Uptrend
        if uptrend_range:
            start, end = uptrend_range
            # Slice for plotting the line segment in color
            mask = (price_full.index >= start) & (price_full.index <= end)
            ax.plot(price_full[mask].index, price_full[mask], color='green', linewidth=2, label='Selected Uptrend')
            # Add shading
            ax.axvspan(start, end, color='green', alpha=0.1)
            
        # Highlight Downtrend
        if downtrend_range:
            start, end = downtrend_range
            mask = (price_full.index >= start) & (price_full.index <= end)
            ax.plot(price_full[mask].index, price_full[mask], color='red', linewidth=2, label='Selected Downtrend')
            ax.axvspan(start, end, color='red', alpha=0.1)
            
        ax.set_title(f"{m['label']} - Selected Trend Periods Context", fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format Y-axis
        from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        # Use the smart density we liked before
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
        ax.ticklabel_format(style='plain', axis='y')
        
        # Format X-axis
        ax.xaxis.set_major_locator(mdates.YearLocator(2)) # Every 2 years for less clutter on long history
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()
        
        output_file = f"context_{market_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close(fig)

if __name__ == "__main__":
    plot_manual_context()
