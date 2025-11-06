#!/usr/bin/env python3
"""
Aggregate results from multiple trials and compute statistics.

Usage:
  python scripts/aggregate_results.py
  python scripts/aggregate_results.py --input results/metrics.csv --output results/aggregated_metrics.csv
"""
import argparse
import pandas as pd
import numpy as np
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def aggregate_results(input_csv, output_csv, min_trials=3):
    """
    Aggregate results from multiple trials per configuration.
    
    Args:
        input_csv: Path to raw metrics CSV
        output_csv: Path to save aggregated metrics
        min_trials: Minimum number of trials required to include a config
    """
    df = pd.read_csv(input_csv)
    
    # Define grouping columns (configuration identifiers)
    group_cols = ['model', 'compression', 'prune_ratio', 'quant_level']
    # Include dataset columns if present to avoid mixing results across datasets
    for extra_col in ['perplexity_dataset', 'perplexity_split']:
        if extra_col in df.columns:
            group_cols.append(extra_col)
    
    # Define metric columns to aggregate
    metric_cols = [
        'perplexity',
        'inference_speed',
        'memory_footprint_mb',
        'total_params',
        'nonzero_params',
        'sparsity',
        'model_size_mb',
    ]
    
    # Drop legacy extrinsic/sentiment columns if present (not used in aggregation)
    legacy_drop = ['sentiment_accuracy', 'sentiment_f1', 'sentiment_precision', 'sentiment_recall',
                   'downstream_dataset', 'downstream_split', 'sentiment_evaluated', 'extrinsic_num_samples']
    for c in legacy_drop:
        if c in df.columns:
            df = df.drop(columns=[c])
    
    # Filter out rows with pruning_unstable=True
    if 'pruning_unstable' in df.columns:
        stable_df = df[df['pruning_unstable'] != True].copy()
        unstable_count = len(df) - len(stable_df)
        if unstable_count > 0:
            print(f"Filtered out {unstable_count} unstable runs")
        df = stable_df
    
    # Group by configuration
    grouped = df.groupby(group_cols, dropna=False)
    
    aggregated_rows = []
    for name, group in grouped:
        n_trials = len(group)
        
        # Skip configs with too few trials
        if n_trials < min_trials:
            print(f"Skipping {name}: only {n_trials} trials (minimum {min_trials} required)")
            continue
        
        row = {col: name[i] for i, col in enumerate(group_cols)}
        row['n_trials'] = n_trials
        
        # Compute statistics for each metric
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            
            values = group[metric].dropna()
            if len(values) == 0:
                row[f'{metric}_mean'] = None
                row[f'{metric}_median'] = None
                row[f'{metric}_std'] = None
                row[f'{metric}_min'] = None
                row[f'{metric}_max'] = None
                row[f'{metric}_iqr'] = None
                continue
            
            row[f'{metric}_mean'] = values.mean()
            row[f'{metric}_median'] = values.median()
            row[f'{metric}_std'] = values.std()
            row[f'{metric}_min'] = values.min()
            row[f'{metric}_max'] = values.max()
            
            # Interquartile range
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            row[f'{metric}_iqr'] = q75 - q25
        
        # Add metadata from first run in group
        first_run = group.iloc[0]
        row['eval_device'] = first_run.get('eval_device', None)
        row['torch_version'] = first_run.get('torch_version', None)
        
        aggregated_rows.append(row)
    
    # Create aggregated dataframe
    agg_df = pd.DataFrame(aggregated_rows)
    
    # Sort by model and compression type (and dataset columns when present)
    if len(agg_df) > 0:
        sort_cols = [c for c in ['model', 'compression', 'prune_ratio', 'quant_level', 'perplexity_dataset', 'perplexity_split'] if c in agg_df.columns]
        agg_df = agg_df.sort_values(sort_cols)
    
    # Save to CSV
    agg_df.to_csv(output_csv, index=False)
    print(f"\nAggregated {len(agg_df)} configurations from {len(df)} total runs")
    print(f"Results saved to: {output_csv}")
    
    # Print summary statistics (guard for empty aggregation)
    print("\n=== Summary Statistics ===")
    print(f"Total configurations: {len(agg_df)}")
    if len(agg_df) == 0:
        print("No configurations met the minimum trials threshold; skipping detailed summary.")
        return agg_df
    print(f"Average trials per config: {agg_df['n_trials'].mean():.1f}")
    print(f"\nPerplexity ranges:")
    if 'perplexity_median' in agg_df.columns:
        print(f"  Min median: {agg_df['perplexity_median'].min():.2f}")
        print(f"  Max median: {agg_df['perplexity_median'].max():.2f}")
    
    if 'inference_speed_median' in agg_df.columns:
        print(f"\nInference speed ranges (tokens/sec):")
        print(f"  Min median: {agg_df['inference_speed_median'].min():.2f}")
        print(f"  Max median: {agg_df['inference_speed_median'].max():.2f}")
    
    if 'model_size_mb_median' in agg_df.columns:
        print(f"\nModel size ranges (MB):")
        print(f"  Min median: {agg_df['model_size_mb_median'].min():.2f}")
        print(f"  Max median: {agg_df['model_size_mb_median'].max():.2f}")
    
    # Identify best configurations
    print("\n=== Best Configurations ===")
    
    if 'perplexity_median' in agg_df.columns and len(agg_df) > 0:
        # Best quality (lowest perplexity)
        best_quality = agg_df.nsmallest(3, 'perplexity_median')[['model', 'compression', 'prune_ratio', 'quant_level', 'perplexity_median', 'model_size_mb_median']]
        print("\nBest Quality (Lowest Perplexity):")
        print(best_quality.to_string(index=False))
        
        # Best compression (smallest size with reasonable quality)
        reasonable_quality = agg_df[agg_df['perplexity_median'] < agg_df['perplexity_median'].quantile(0.5)]
        if len(reasonable_quality) > 0 and 'model_size_mb_median' in reasonable_quality.columns:
            best_compression = reasonable_quality.nsmallest(3, 'model_size_mb_median')[['model', 'compression', 'prune_ratio', 'quant_level', 'perplexity_median', 'model_size_mb_median']]
            print("\nBest Compression (Smallest Size, Reasonable Quality):")
            print(best_compression.to_string(index=False))
        
        # Best speed
        if 'inference_speed_median' in agg_df.columns:
            best_speed = agg_df.nlargest(3, 'inference_speed_median')[['model', 'compression', 'prune_ratio', 'quant_level', 'inference_speed_median', 'perplexity_median']]
            print("\nBest Speed (Highest Tokens/Sec):")
            print(best_speed.to_string(index=False))
    
    return agg_df


def main():
    parser = argparse.ArgumentParser(description='Aggregate experimental results across multiple trials')
    parser.add_argument('--input', type=str, default=os.path.join(ROOT, 'results', 'metrics.csv'),
                        help='Input CSV with raw metrics')
    parser.add_argument('--output', type=str, default=os.path.join(ROOT, 'results', 'aggregated_metrics.csv'),
                        help='Output CSV for aggregated metrics')
    parser.add_argument('--min-trials', type=int, default=3,
                        help='Minimum number of trials required to include a configuration')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    aggregate_results(args.input, args.output, args.min_trials)


if __name__ == '__main__':
    main()
