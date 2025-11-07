#!/usr/bin/env python3
"""
Results summarization for Microscale LLM Compression Experiments

This script now ONLY generates summary statistics (summary_stats.txt).
All plotting functionality has been removed to simplify outputs.
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def load_results(csv_path):
    """Load experiment results from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiments from {csv_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nUnique models: {df['model'].unique()}")
    print(f"Unique compressions: {df['compression'].unique()}")
    return df

# Plotting functions removed

# Plotting functions removed

# Plotting functions removed

# Plotting functions removed

# Plotting functions removed

def generate_summary_stats(df, output_dir):
    """Generate summary statistics table."""
    summary = df.groupby('compression').agg({
        'perplexity': ['mean', 'std', 'min', 'max'],
        'inference_speed': ['mean', 'std', 'min', 'max'],
        'model_size_mb': ['mean', 'std', 'min', 'max'],
        'sparsity': ['mean', 'std'],
        'memory_footprint_mb': ['mean', 'std']
    }).round(2)
    
    output_path = output_dir / 'summary_stats.txt'
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MICROSCALE LLM COMPRESSION - SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
        
        # Count unstable runs
        if 'pruning_unstable' in df.columns:
            unstable_count = df['pruning_unstable'].sum()
            f.write(f"Unstable pruning runs (NaN detected): {unstable_count}\n")
        
        # Count by device
        if 'eval_device' in df.columns:
            device_counts = df['eval_device'].value_counts()
            f.write("\nExperiments by device:\n")
            f.write(device_counts.to_string())
            f.write("\n")
    
    print(f"Saved summary statistics to {output_path}")
    print("\nSummary Statistics:")
    print(summary)

def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from microscale LLM experiment results."
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='results/metrics.csv',
        help='Path to metrics CSV file (default: results/metrics.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for plots (default: results/)'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    print("=" * 80)
    print("MICROSCALE LLM COMPRESSION - SUMMARY")
    print("=" * 80)
    
    # Load data
    df = load_results(csv_path)
    
    # Generate summary statistics
    generate_summary_stats(df, output_dir)
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPLETE")
    print("=" * 80)
    print(f"Summary saved to {output_dir}/summary_stats.txt")

if __name__ == '__main__':
    main()
