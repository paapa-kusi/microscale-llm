#!/usr/bin/env python3
"""
Plot Results - Visualization Script for Microscale LLM Compression Experiments

Generates comprehensive plots from experiment results:
- Pareto frontier: inference speed vs perplexity
- Trade-off plots: model size vs perplexity
- Heatmaps: pruning ratio × quantization level → perplexity
- Sparsity analysis
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(csv_path):
    """Load experiment results from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiments from {csv_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nUnique models: {df['model'].unique()}")
    print(f"Unique compressions: {df['compression'].unique()}")
    return df

def plot_pareto_frontier(df, output_dir):
    """
    Plot Pareto frontier: inference_speed vs perplexity.
    Shows trade-off between model performance and inference efficiency.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out unstable/failed runs
    df_stable = df[df['perplexity'] < 1000].copy()  # Remove catastrophic failures
    
    # Create scatter plot with compression type as color
    for compression in df_stable['compression'].unique():
        subset = df_stable[df_stable['compression'] == compression]
        ax.scatter(subset['inference_speed'], subset['perplexity'], 
                  label=compression, alpha=0.7, s=100, edgecolors='black', linewidths=0.5)
    
    # Identify Pareto-optimal points (lower perplexity, higher speed is better)
    pareto_mask = []
    for idx, row in df_stable.iterrows():
        is_pareto = True
        for idx2, row2 in df_stable.iterrows():
            if idx == idx2:
                continue
            # Point is dominated if another point has both better speed and better perplexity
            if row2['inference_speed'] > row['inference_speed'] and row2['perplexity'] < row['perplexity']:
                is_pareto = False
                break
        pareto_mask.append(is_pareto)
    
    pareto_df = df_stable[pareto_mask].sort_values('inference_speed')
    if len(pareto_df) > 0:
        ax.plot(pareto_df['inference_speed'], pareto_df['perplexity'], 
               'r--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    ax.set_xlabel('Inference Speed (samples/sec)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Pareto Frontier: Speed vs Quality Trade-off', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'pareto_frontier.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pareto frontier plot to {output_path}")
    plt.close()

def plot_size_vs_perplexity(df, output_dir):
    """
    Plot model size vs perplexity.
    Shows compression efficiency trade-off.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_stable = df[(df['perplexity'] < 1000) & (df['model_size_mb'].notna())].copy()
    
    for compression in df_stable['compression'].unique():
        subset = df_stable[df_stable['compression'] == compression]
        ax.scatter(subset['model_size_mb'], subset['perplexity'], 
                  label=compression, alpha=0.7, s=100, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Model Size (MB)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Compression Trade-off: Size vs Quality', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'size_vs_perplexity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved size vs perplexity plot to {output_path}")
    plt.close()

def plot_pruning_heatmap(df, output_dir):
    """
    Plot heatmap: pruning ratio × quantization level → perplexity.
    Only for combined compression experiments.
    """
    df_combined = df[(df['compression'] == 'combined') & 
                     (df['prune_ratio'].notna()) & 
                     (df['quant_level'].notna())].copy()
    
    if len(df_combined) == 0:
        print("No combined compression experiments found, skipping heatmap.")
        return
    
    # Create pivot table
    pivot = df_combined.pivot_table(
        values='perplexity', 
        index='prune_ratio', 
        columns='quant_level',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Log-scale perplexity for better color distribution
    pivot_log = np.log10(pivot.clip(lower=1))  # Avoid log(0)
    
    sns.heatmap(pivot_log, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'log10(Perplexity)'}, ax=ax)
    
    ax.set_xlabel('Quantization Level', fontsize=12)
    ax.set_ylabel('Pruning Ratio', fontsize=12)
    ax.set_title('Combined Compression: Pruning × Quantization → Perplexity', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'pruning_quant_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved pruning×quantization heatmap to {output_path}")
    plt.close()

def plot_sparsity_analysis(df, output_dir):
    """
    Plot sparsity vs perplexity for pruned models.
    Shows relationship between sparsity and model quality.
    """
    df_pruned = df[(df['compression'].isin(['pruning', 'combined'])) & 
                   (df['sparsity'].notna()) & 
                   (df['perplexity'] < 1000)].copy()
    
    if len(df_pruned) == 0:
        print("No pruning experiments with sparsity data found, skipping sparsity analysis.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by pruning stability
    colors = ['red' if unstable else 'green' 
              for unstable in df_pruned['pruning_unstable'].fillna(False)]
    
    ax.scatter(df_pruned['sparsity'], df_pruned['perplexity'], 
              c=colors, alpha=0.7, s=100, edgecolors='black', linewidths=0.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=10, label='Stable', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Unstable (NaN detected)', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_xlabel('Sparsity (fraction of zero weights)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Pruning Analysis: Sparsity vs Quality', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'sparsity_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sparsity analysis plot to {output_path}")
    plt.close()

def plot_device_comparison(df, output_dir):
    """
    Plot inference speed comparison across devices (CPU vs GPU).
    Shows hardware efficiency differences.
    """
    if 'eval_device' not in df.columns or df['eval_device'].isna().all():
        print("No eval_device data found, skipping device comparison.")
        return
    
    df_valid = df[(df['inference_speed'].notna()) & 
                  (df['eval_device'].notna()) & 
                  (df['inference_speed'] > 0)].copy()
    
    if len(df_valid) == 0:
        print("No valid device data for comparison, skipping.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by compression type and device
    df_valid['compression_device'] = df_valid['compression'] + ' (' + df_valid['eval_device'] + ')'
    
    grouped = df_valid.groupby('compression_device')['inference_speed'].mean().sort_values()
    
    grouped.plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
    
    ax.set_xlabel('Inference Speed (samples/sec)', fontsize=12)
    ax.set_ylabel('Compression Type (Device)', fontsize=12)
    ax.set_title('Device Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / 'device_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved device comparison plot to {output_path}")
    plt.close()

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
    print("MICROSCALE LLM COMPRESSION - RESULTS VISUALIZATION")
    print("=" * 80)
    
    # Load data
    df = load_results(csv_path)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_pareto_frontier(df, output_dir)
    plot_size_vs_perplexity(df, output_dir)
    plot_pruning_heatmap(df, output_dir)
    plot_sparsity_analysis(df, output_dir)
    plot_device_comparison(df, output_dir)
    
    # Generate summary statistics
    generate_summary_stats(df, output_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"All plots saved to {output_dir}/")

if __name__ == '__main__':
    main()
