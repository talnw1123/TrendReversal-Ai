import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob

def plot_reversals():
    # 1. Find Reversal Points File
    # User mentioned result/reversal_points.csv, but we saved to root. Check both.
    possible_paths = [
        "result/reversal_points.csv",
        "reversal_points.csv"
    ]
    
    points_file = None
    for p in possible_paths:
        if os.path.exists(p):
            points_file = p
            break
            
    if not points_file:
        print("Error: reversal_points.csv not found.")
        return

    print(f"Loading reversal points from {points_file}...")
    df_points = pd.read_csv(points_file, parse_dates=['Date'])
    
    # 2. Group by Market and Trend Type
    markets = df_points['Market'].unique()
    trend_types = ['uptrend', 'downtrend']
    
    test_data_dir = "trend_data_manual/split/test"
    
    for market in markets:
        for trend in trend_types:
            print(f"Plotting {market} ({trend})...")
            
            # 3. Load Specific Test Data
            # Find the specific file (e.g., US_uptrend_labeled.csv)
            # Note: The file might be in a subdir like 'label' or directly in 'test'
            file_pattern = os.path.join(test_data_dir, f"**/{market}_{trend}_labeled.csv")
            found_files = glob.glob(file_pattern, recursive=True)
            
            if not found_files:
                print(f"  Warning: No test data found for {market} {trend}. Skipping.")
                continue
                
            # Should be only one file per market+trend in test set
            target_file = found_files[0]
            df_market = pd.read_csv(target_file, index_col=0, parse_dates=True).sort_index()
            
            if df_market.empty:
                continue
            
            # 4. Plot
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot Price Line
            ax.plot(df_market.index, df_market['Close'], color='lightgray', linewidth=1.5, label='Price (Test Set)')
            
            # Filter Points by Date Range of this file
            start_date = df_market.index.min()
            end_date = df_market.index.max()
            
            mask = (df_points['Market'] == market) & \
                   (df_points['Date'] >= start_date) & \
                   (df_points['Date'] <= end_date)
                   
            period_points = df_points[mask]
            
            # Plot URP
            urp = period_points[period_points['Type'].str.contains('URP')]
            if not urp.empty:
                ax.scatter(urp['Date'], urp['Price'], color='green', marker='^', s=100, label='Predicted URP', zorder=5)
                
            # Plot DRP
            drp = period_points[period_points['Type'].str.contains('DRP')]
            if not drp.empty:
                ax.scatter(drp['Date'], drp['Price'], color='red', marker='v', s=100, label='Predicted DRP', zorder=5)
                
            ax.set_title(f"{market} {trend.capitalize()} - Predicted Reversal Points", fontsize=14)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Format Y-axis
            from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
            ax.ticklabel_format(style='plain', axis='y')
            
            # Format X-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            
            output_file = f"prediction_{market}_{trend}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  Saved {output_file}")
            plt.close(fig)

if __name__ == "__main__":
    plot_reversals()
