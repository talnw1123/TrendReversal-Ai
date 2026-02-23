import pandas as pd
import numpy as np
import os
import glob

def apply_triple_barrier(df, t=20, h=0.02):
    """
    Applies Triple Barrier Method to label data.
    t: Time horizon (bars)
    h: Barrier threshold (percentage, e.g., 0.02 for 2%)
    """
    labels = []
    
    # Pre-calculate barriers for speed
    # Note: This is a simplified vectorizable approach or iterative. 
    # For clarity and correctness with High/Low, iterative is safer for now.
    
    close_prices = df['Close'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    n = len(df)
    
    for i in range(n):
        # End of dataset handling
        if i + t >= n:
            labels.append(0) # Or NaN, but 0 (no reversal) is safer for now
            continue
            
        current_close = close_prices[i]
        upper_barrier = current_close * (1 + h)
        lower_barrier = current_close * (1 - h)
        
        # Look ahead window
        window_highs = high_prices[i+1 : i+1+t]
        window_lows = low_prices[i+1 : i+1+t]
        
        # Find first breach
        # We need to know WHICH barrier was hit FIRST.
        # This is tricky with just min/max. We iterate the window.
        label = 0
        for j in range(len(window_highs)):
            h_price = window_highs[j]
            l_price = window_lows[j]
            
            hit_upper = h_price >= upper_barrier
            hit_lower = l_price <= lower_barrier
            
            if hit_upper and hit_lower:
                # Both hit in same bar? Rare but possible. 
                # Usually prioritize the one that matches the close direction or treat as 0 (volatility).
                # For reversal prediction:
                # If we are looking for reversals, maybe we care about the FIRST one.
                # Let's assume Close determines the winner if both hit, or just take the first check.
                # Let's prioritize Upper (+1) for simplicity or check Close.
                label = 1 
                break
            elif hit_upper:
                label = 1
                break
            elif hit_lower:
                label = -1
                break
                
        labels.append(label)
        
    return labels

def add_swing_features(df, lookback=5):
    """Adds Swing High/Low features based on user snippet."""
    df['swing_high'] = (df['High'] == df['High'].rolling(2*lookback+1, center=True).max())
    df['swing_low'] = (df['Low'] == df['Low'].rolling(2*lookback+1, center=True).min())
    # Fill NaNs resulting from rolling with False
    df['swing_high'] = df['swing_high'].fillna(False)
    df['swing_low'] = df['swing_low'].fillna(False)
    return df

def label_data():
    base_dir = "trend_data_manual/split"
    splits = ['train', 'val', 'test']
    
    print("Labeling data with Triple Barrier Method (T=20, H=2%)...")
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        csv_files = glob.glob(os.path.join(split_dir, "*.csv"))
        
        for file_path in csv_files:
            # Skip already labeled files
            if "_labeled.csv" in file_path:
                continue
                
            print(f"  Processing {split}/{os.path.basename(file_path)}...")
            
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # 1. Add Swing Features
            df = add_swing_features(df)
            
            # 2. Apply Triple Barrier Labeling
            # Using T=20, H=0.02 (2%)
            df['label'] = apply_triple_barrier(df, t=20, h=0.02)
            
            # Save
            output_file = file_path.replace(".csv", "_labeled.csv")
            df.to_csv(output_file)
            
            # Stats
            counts = df['label'].value_counts()
            print(f"    Labels: {counts.to_dict()}")

    print("Labeling complete.")

if __name__ == "__main__":
    label_data()
