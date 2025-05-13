#!/usr/bin/env python3
"""
Clean the bpRNA dataset by removing sequences with ambiguous nucleotides.
Only keeps sequences containing standard nucleotides (A, U, C, G).
"""
import os
import csv
import pandas as pd
import time

def clean_dataset(input_file, output_file=None):
    """
    Remove sequences with ambiguous nucleotides from a dataset.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (default: auto-generated)
        
    Returns:
        tuple: (total_sequences, clean_sequences, removed_sequences)
    """
    if output_file is None:
        # Generate output filename from input filename
        base_name = os.path.basename(input_file)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(os.path.dirname(input_file), f"{name}_clean{ext}")
    
    print(f"Cleaning dataset: {input_file}")
    start_time = time.time()

    # Try to load with pandas for efficiency
    try:
        df = pd.read_csv(input_file)
        total_sequences = len(df)
        print(f"Loaded {total_sequences} sequences with pandas")
        
        # Get indices of rows with clean sequences (only A, U, C, G)
        clean_indices = []
        removed_indices = []
        
        for idx, row in df.iterrows():
            sequence = row['sequence']
            # Check if sequence contains only standard nucleotides
            if all(c in 'AUCGaucg' for c in sequence):
                clean_indices.append(idx)
            else:
                removed_indices.append(idx)
            
            # Print progress
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1}/{total_sequences} sequences")
        
        # Create a new DataFrame with only clean sequences
        clean_df = df.loc[clean_indices].copy()
        clean_sequences = len(clean_df)
        removed_sequences = len(removed_indices)
        
        # Save clean dataset
        clean_df.to_csv(output_file, index=False)
        
    except Exception as e:
        print(f"Error using pandas: {e}")
        print("Falling back to CSV reader/writer...")
        
        # Fallback to CSV reader/writer for large files
        total_sequences = 0
        clean_sequences = 0
        removed_sequences = 0
        
        with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Write header
            header = next(reader)
            writer.writerow(header)
            
            # Find sequence index
            try:
                seq_idx = header.index('sequence')
            except ValueError:
                print("Error: 'sequence' column not found in header")
                return 0, 0, 0
            
            # Process each row
            for i, row in enumerate(reader):
                total_sequences += 1
                
                if len(row) > seq_idx:
                    sequence = row[seq_idx]
                    
                    # Check if sequence contains only standard nucleotides
                    if all(c in 'AUCGaucg' for c in sequence):
                        writer.writerow(row)
                        clean_sequences += 1
                    else:
                        removed_sequences += 1
                
                # Print progress
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i + 1} sequences")
    
    # Calculate percentage and time
    removed_percentage = (removed_sequences / total_sequences) * 100 if total_sequences > 0 else 0
    elapsed_time = time.time() - start_time
    
    print(f"\nResults for {os.path.basename(input_file)}:")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Clean sequences: {clean_sequences} ({100 - removed_percentage:.2f}%)")
    print(f"  Removed sequences: {removed_sequences} ({removed_percentage:.2f}%)")
    print(f"  Clean dataset saved to: {output_file}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    
    return total_sequences, clean_sequences, removed_sequences

if __name__ == "__main__":
    # Define paths to bpRNA files
    data_dir = "data/bprna"
    files = [
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "valid.csv") if os.path.exists(os.path.join(data_dir, "valid.csv")) else None,
        os.path.join(data_dir, "test.csv") if os.path.exists(os.path.join(data_dir, "test.csv")) else None
    ]
    
    # Track overall stats
    total_overall = 0
    clean_overall = 0
    removed_overall = 0
    
    # Process each file if it exists
    for file_path in files:
        if file_path and os.path.exists(file_path):
            print("\n" + "="*50)
            print(f"Processing file: {file_path}")
            print("="*50)
            
            total, clean, removed = clean_dataset(file_path)
            
            total_overall += total
            clean_overall += clean
            removed_overall += removed
        else:
            if file_path:
                print(f"File not found: {file_path}")
    
    # Print overall stats
    if total_overall > 0:
        removed_percentage = (removed_overall / total_overall) * 100
        
        print("\n" + "="*50)
        print("OVERALL RESULTS:")
        print("="*50)
        print(f"Total sequences: {total_overall}")
        print(f"Clean sequences: {clean_overall} ({100 - removed_percentage:.2f}%)")
        print(f"Removed sequences: {removed_overall} ({removed_percentage:.2f}%)") 