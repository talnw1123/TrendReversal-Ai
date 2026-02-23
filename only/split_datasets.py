import pandas as pd
import os
import glob

def split_datasets():
    input_dir = "trend_data_manual"
    output_base = os.path.join(input_dir, "split")
    
    # Create output directories
    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base, split_name), exist_ok=True)
        
    print(f"Splitting datasets in '{input_dir}' into Train/Val/Test...")
    
    # Find all CSV files in the input directory (excluding subdirectories)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        market_trend_name = os.path.splitext(filename)[0]
        
        print(f"Processing {filename}...")
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.sort_index(inplace=True) # Ensure time order
        
        n = len(df)
        if n < 10:
            print(f"  Warning: {filename} has too few rows ({n}). Skipping.")
            continue
            
        # Sequential Split
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        # Save splits
        train_df.to_csv(os.path.join(output_base, 'train', filename))
        val_df.to_csv(os.path.join(output_base, 'val', filename))
        test_df.to_csv(os.path.join(output_base, 'test', filename))
        
        print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("Splitting complete.")

if __name__ == "__main__":
    split_datasets()
