#!/usr/bin/env python3
"""
Enhanced Analysis and Visualization for Microscale LLM Compression
Focuses on understanding patterns between perplexity and compression methods
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)

def load_and_prepare_data(csv_path):
    """Load and prepare the metrics data."""
    df = pd.read_csv(csv_path)
    
    # Filter out EleutherAI model
    df = df[~df['model'].str.contains('EleutherAI', case=False, na=False)]
    
    # Fill missing compression values with 'baseline'
    df['compression'] = df['compression'].fillna('baseline')
    
    # Create a more descriptive compression label
    df['compression_label'] = df.apply(lambda row: 
        'Baseline' if pd.isna(row['compression']) or row['compression'] == '' 
        else f"Pruning {row['prune_ratio']}" if row['compression'] == 'pruning' 
        else f"Quant {row['quant_level']}" if row['compression'] == 'quantization'
        else f"Prune {row['prune_ratio']} + Quant {row['quant_level']}" if row['compression'] == 'combined'
        else 'Baseline', axis=1)
    
    print(f"Loaded {len(df)} experiments")
    print(f"Models: {df['model'].unique()}")
    print(f"Compression types: {df['compression'].unique()}")
    
    return df

def plot_perplexity_by_model_and_method(df, output_dir):
    """
    Create grouped bar plots showing perplexity for each model and compression method.
    Note: Excludes extreme pruning (0.9) for better visualization due to catastrophic failure.
    """
    # Filter out extreme pruning (0.9) for clearer visualization
    df_filtered = df[~((df['compression'] == 'pruning') & (df['prune_ratio'] == 0.9)) &
                     ~((df['compression'] == 'combined') & (df['prune_ratio'] == 0.9))].copy()
    
    # Prepare data - average across seeds (dropna=False to handle NaN in prune_ratio/quant_level)
    df_avg = df_filtered.groupby(['model', 'compression', 'prune_ratio', 'quant_level'], dropna=False).agg({
        'perplexity': 'mean',
        'model_size_mb': 'mean',
        'inference_speed': 'mean'
    }).reset_index()
    
    # Create compression method label
    df_avg['method'] = df_avg.apply(lambda row: 
        'Baseline' if row['compression'] == 'baseline'
        else f"Prune {row['prune_ratio']:.1f}" if row['compression'] == 'pruning'
        else f"{row['quant_level']}" if row['compression'] == 'quantization'
        else f"P{row['prune_ratio']:.1f}+{row['quant_level']}" if row['compression'] == 'combined'
        else 'Unknown', axis=1)
    
    # Sort models by size: small -> medium -> large
    models = ['gpt2', 'gpt2-medium', 'gpt2-large']
    
    # Define method order for consistent display (includes combined methods now)
    method_order = ['Baseline', 'Prune 0.1', 'Prune 0.5', 'INT8', 'INT4', 
                    'P0.1+INT8', 'P0.1+INT4', 'P0.5+INT8', 'P0.5+INT4']
    all_methods = [m for m in method_order if m in df_avg['method'].values]
    
    if len(all_methods) == 0:
        print("⚠ Warning: No methods found after filtering. Check data.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    x = np.arange(len(models))
    n_methods = len(all_methods)
    width = 0.8 / n_methods
    
    # Use a neutral, professional color palette (Set2)
    colors = sns.color_palette("Set2", n_methods)
    
    for idx, method in enumerate(all_methods):
        perplexities = []
        for model in models:
            model_data = df_avg[(df_avg['model'] == model) & (df_avg['method'] == method)]
            if len(model_data) > 0:
                val = model_data['perplexity'].values[0]
                perplexities.append(val)
            else:
                perplexities.append(np.nan)
        
        offset = (idx - n_methods/2 + 0.5) * width
        bars = ax.bar(x + offset, perplexities, width, label=method, color=colors[idx], edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars with actual perplexity value
        for i, (bar, val) in enumerate(zip(bars, perplexities)):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Perplexity', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Perplexity Comparison: All Compression Methods\n(Excludes: Prune 0.9 due to catastrophic failure)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['GPT-2\n(124M)', 'GPT-2-Medium\n(355M)', 'GPT-2-Large\n(774M)'], fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=True, shadow=True)
    # Use LINEAR scale instead of log to show actual differences
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'perplexity_by_model_method.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved perplexity by model/method plot to {output_path}")
    plt.close()

def plot_pruning_effect(df, output_dir):
    """
    Show the effect of different pruning ratios on perplexity for each model.
    """
    df_pruning = df[df['compression'].isin(['baseline', 'pruning'])].copy()
    df_pruning['prune_ratio'] = df_pruning['prune_ratio'].fillna(0.0)
    
    df_avg = df_pruning.groupby(['model', 'prune_ratio']).agg({
        'perplexity': ['mean', 'std'],
        'sparsity': 'mean'
    }).reset_index()
    
    df_avg.columns = ['model', 'prune_ratio', 'perplexity_mean', 'perplexity_std', 'sparsity']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Perplexity vs Pruning Ratio
    ax1 = axes[0]
    for model in df_avg['model'].unique():
        model_data = df_avg[df_avg['model'] == model].sort_values('prune_ratio')
        ax1.plot(model_data['prune_ratio'], model_data['perplexity_mean'], 
                marker='o', linewidth=2.5, markersize=10, label=model)
        ax1.fill_between(model_data['prune_ratio'], 
                        model_data['perplexity_mean'] - model_data['perplexity_std'],
                        model_data['perplexity_mean'] + model_data['perplexity_std'],
                        alpha=0.2)
    
    ax1.set_xlabel('Pruning Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Pruning on Model Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Perplexity vs Sparsity
    ax2 = axes[1]
    for model in df_avg['model'].unique():
        model_data = df_avg[df_avg['model'] == model].sort_values('sparsity')
        ax2.plot(model_data['sparsity'] * 100, model_data['perplexity_mean'], 
                marker='s', linewidth=2.5, markersize=10, label=model)
    
    ax2.set_xlabel('Sparsity (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax2.set_title('Perplexity vs Model Sparsity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'pruning_effect_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved pruning effect analysis to {output_path}")
    plt.close()

def plot_quantization_effect(df, output_dir):
    """
    Show the effect of quantization on perplexity and model size for each model.
    """
    df_quant = df[df['compression'].isin(['baseline', 'quantization'])].copy()
    
    # Set quant_level to FP16 for baseline before grouping
    df_quant.loc[df_quant['compression'] == 'baseline', 'quant_level'] = 'FP16'
    
    df_avg = df_quant.groupby(['model', 'quant_level']).agg({
        'perplexity': ['mean', 'std'],
        'model_size_mb': 'mean',
        'inference_speed': 'mean'
    }).reset_index()
    
    df_avg.columns = ['model', 'quant_level', 'perplexity_mean', 'perplexity_std', 
                      'model_size_mb', 'inference_speed']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Perplexity by Quantization Level
    ax1 = axes[0]
    # Sort models by size: small -> medium -> large
    models = ['gpt2', 'gpt2-medium', 'gpt2-large']
    quant_levels = ['FP16', 'INT8', 'INT4']
    x = np.arange(len(models))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, ql in enumerate(quant_levels):
        perplexities = []
        errors = []
        for model in models:
            model_data = df_avg[(df_avg['model'] == model) & (df_avg['quant_level'] == ql)]
            if len(model_data) > 0:
                perplexities.append(model_data['perplexity_mean'].values[0])
                errors.append(model_data['perplexity_std'].values[0])
            else:
                perplexities.append(np.nan)
                errors.append(0)
        
        offset = (idx - 1) * width
        ax1.bar(x + offset, perplexities, width, label=ql, color=colors[idx], 
               yerr=errors, capsize=5)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Quantization on Perplexity', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['GPT-2\n(124M)', 'GPT-2-Medium\n(355M)', 'GPT-2-Large\n(774M)'])
    ax1.legend(fontsize=11, title='Quantization Level')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Model Size Reduction
    ax2 = axes[1]
    for idx, ql in enumerate(quant_levels):
        sizes = []
        for model in models:
            model_data = df_avg[(df_avg['model'] == model) & (df_avg['quant_level'] == ql)]
            if len(model_data) > 0:
                sizes.append(model_data['model_size_mb'].values[0])
            else:
                sizes.append(np.nan)
        
        offset = (idx - 1) * width
        ax2.bar(x + offset, sizes, width, label=ql, color=colors[idx])
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Size by Quantization Level', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['GPT-2\n(124M)', 'GPT-2-Medium\n(355M)', 'GPT-2-Large\n(774M)'])
    ax2.legend(fontsize=11, title='Quantization Level')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'quantization_effect_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved quantization effect analysis to {output_path}")
    plt.close()

def plot_combined_effects(df, output_dir):
    """
    Show the combined effects of pruning and quantization.
    """
    df_all = df[df['compression'].isin(['baseline', 'pruning', 'quantization', 'combined'])].copy()
    
    # Filter out catastrophic pruning failures (>10000 perplexity) to avoid skewing averages
    df_all = df_all[~((df_all['compression'] == 'pruning') & (df_all['perplexity'] > 10000))]
    
    # Create a simplified label
    df_all['simple_method'] = df_all.apply(lambda row:
        'Baseline' if row['compression'] == 'baseline'
        else 'Pruning Only' if row['compression'] == 'pruning'
        else 'Quantization Only' if row['compression'] == 'quantization'
        else 'Combined', axis=1)
    
    df_avg = df_all.groupby(['model', 'simple_method']).agg({
        'perplexity': 'mean',
        'model_size_mb': 'mean',
        'inference_speed': 'mean'
    }).reset_index()
    
    # Create 1x3 layout (removed memory footprint)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Perplexity comparison
    ax1 = axes[0]
    methods = ['Baseline', 'Pruning Only', 'Quantization Only', 'Combined']
    # Sort models by size: small -> medium -> large
    models = ['gpt2', 'gpt2-medium', 'gpt2-large']
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for idx, method in enumerate(methods):
        perps = []
        for model in models:
            model_data = df_avg[(df_avg['model'] == model) & (df_avg['simple_method'] == method)]
            if len(model_data) > 0:
                perps.append(model_data['perplexity'].values[0])
            else:
                perps.append(0)
        ax1.bar(x + idx * width - width*1.5, perps, width, label=method, color=colors[idx])
    
    ax1.set_ylabel('Perplexity (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Perplexity Comparison Across Methods\n(Pruning = avg of 10% & 50%; excludes catastrophic 90%)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['GPT-2\n(124M)', 'GPT-2-Medium\n(355M)', 'GPT-2-Large\n(774M)'], fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Model size comparison
    ax2 = axes[1]
    for idx, method in enumerate(methods):
        sizes = []
        for model in models:
            model_data = df_avg[(df_avg['model'] == model) & (df_avg['simple_method'] == method)]
            if len(model_data) > 0:
                sizes.append(model_data['model_size_mb'].values[0])
            else:
                sizes.append(0)
        ax2.bar(x + idx * width - width*1.5, sizes, width, label=method, color=colors[idx])
    
    ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['GPT-2\n(124M)', 'GPT-2-Medium\n(355M)', 'GPT-2-Large\n(774M)'], fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Inference speed comparison
    ax3 = axes[2]
    for idx, method in enumerate(methods):
        speeds = []
        for model in models:
            model_data = df_avg[(df_avg['model'] == model) & (df_avg['simple_method'] == method)]
            if len(model_data) > 0:
                speeds.append(model_data['inference_speed'].values[0])
            else:
                speeds.append(0)
        ax3.bar(x + idx * width - width*1.5, speeds, width, label=method, color=colors[idx])
    
    ax3.set_ylabel('Inference Speed (samples/sec)', fontsize=12, fontweight='bold')
    ax3.set_title('Inference Speed Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['GPT-2\n(124M)', 'GPT-2-Medium\n(355M)', 'GPT-2-Large\n(774M)'], fontsize=10)
    ax3.legend(fontsize=10)
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'combined_effects_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined effects comparison to {output_path}")
    plt.close()

def plot_compression_tradeoffs(df, output_dir):
    """
    Create scatter plots showing trade-offs between metrics.
    """
    df_avg = df.groupby(['model', 'compression', 'prune_ratio', 'quant_level']).agg({
        'perplexity': 'mean',
        'model_size_mb': 'mean',
        'inference_speed': 'mean'
    }).reset_index()
    
    # Filter out extreme perplexities for better visualization
    df_viz = df_avg[df_avg['perplexity'] < 100000].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Model Size vs Perplexity
    ax1 = axes[0]
    for model in df_viz['model'].unique():
        model_data = df_viz[df_viz['model'] == model]
        for compression in model_data['compression'].unique():
            comp_data = model_data[model_data['compression'] == compression]
            marker_style = 'o' if compression == 'baseline' else 's' if compression == 'pruning' \
                          else '^' if compression == 'quantization' else 'D'
            ax1.scatter(comp_data['model_size_mb'], comp_data['perplexity'], 
                       s=150, alpha=0.7, marker=marker_style,
                       label=f"{model}-{compression}" if len(df_viz['model'].unique()) > 1 else compression)
    
    ax1.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Compression Trade-off: Size vs Performance', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Inference Speed vs Perplexity
    ax2 = axes[1]
    for model in df_viz['model'].unique():
        model_data = df_viz[df_viz['model'] == model]
        for compression in model_data['compression'].unique():
            comp_data = model_data[model_data['compression'] == compression]
            marker_style = 'o' if compression == 'baseline' else 's' if compression == 'pruning' \
                          else '^' if compression == 'quantization' else 'D'
            ax2.scatter(comp_data['inference_speed'], comp_data['perplexity'], 
                       s=150, alpha=0.7, marker=marker_style,
                       label=f"{model}-{compression}" if len(df_viz['model'].unique()) > 1 else compression)
    
    ax2.set_xlabel('Inference Speed (samples/sec)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Trade-off: Speed vs Quality', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'compression_tradeoffs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved compression trade-offs plot to {output_path}")
    plt.close()

def plot_heatmaps_per_model(df, output_dir):
    """
    Create heatmaps showing perplexity for pruning x quantization combinations per model.
    """
    df_combined = df[df['compression'] == 'combined'].copy()
    
    if len(df_combined) == 0:
        print("⚠ No combined compression data found, skipping heatmaps")
        return
    
    models = df_combined['model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(8*len(models), 6))
    
    if len(models) == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_data = df_combined[df_combined['model'] == model].copy()
        
        # Average across seeds
        pivot_data = model_data.groupby(['prune_ratio', 'quant_level'])['perplexity'].mean().reset_index()
        pivot = pivot_data.pivot(index='prune_ratio', columns='quant_level', values='perplexity')
        
        # Use log scale for perplexity
        pivot_log = np.log10(pivot.clip(lower=1))
        
        ax = axes[idx]
        sns.heatmap(pivot_log, annot=pivot, fmt='.1f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'log10(Perplexity)'}, ax=ax,
                   vmin=0, vmax=pivot_log.max().max())
        
        ax.set_xlabel('Quantization Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pruning Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'{model.upper()}\nPruning × Quantization → Perplexity', 
                    fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'pruning_quantization_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved pruning×quantization heatmaps to {output_path}")
    plt.close()

def generate_summary_report(df, output_dir):
    """
    Generate a text summary of key findings.
    """
    output_path = output_dir / 'analysis_summary.txt'
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MICROSCALE LLM COMPRESSION - ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Models tested: {', '.join(df['model'].unique())}\n")
        f.write(f"Compression methods: {', '.join(df['compression'].unique())}\n\n")
        
        # Baseline performance
        f.write("BASELINE PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        baselines = df[df['compression'] == 'baseline'].groupby('model').agg({
            'perplexity': 'mean',
            'model_size_mb': 'mean',
            'inference_speed': 'mean'
        })
        f.write(baselines.to_string() + "\n\n")
        
        # Best compression per model
        f.write("BEST COMPRESSION RESULTS (Lowest Perplexity)\n")
        f.write("-" * 80 + "\n")
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            best = model_data.loc[model_data['perplexity'].idxmin()]
            f.write(f"\n{model}:\n")
            f.write(f"  Compression: {best['compression']}\n")
            if pd.notna(best['prune_ratio']):
                f.write(f"  Prune Ratio: {best['prune_ratio']}\n")
            if pd.notna(best['quant_level']):
                f.write(f"  Quantization: {best['quant_level']}\n")
            f.write(f"  Perplexity: {best['perplexity']:.2f}\n")
            f.write(f"  Model Size: {best['model_size_mb']:.2f} MB\n")
            f.write(f"  Inference Speed: {best['inference_speed']:.2f} samples/sec\n")
        
        f.write("\n")
        
        # Compression effectiveness
        f.write("\nCOMPRESSION EFFECTIVENESS SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        for model in df['model'].unique():
            f.write(f"\n{model.upper()}:\n")
            model_data = df[df['model'] == model]
            baseline_perp = model_data[model_data['compression'] == 'baseline']['perplexity'].mean()
            baseline_size = model_data[model_data['compression'] == 'baseline']['model_size_mb'].mean()
            
            f.write(f"  Baseline: Perplexity={baseline_perp:.2f}, Size={baseline_size:.2f}MB\n")
            
            # Pruning
            for ratio in [0.1, 0.5, 0.9]:
                pruned = model_data[(model_data['compression'] == 'pruning') & 
                                   (model_data['prune_ratio'] == ratio)]
                if len(pruned) > 0:
                    perp = pruned['perplexity'].mean()
                    degradation = ((perp - baseline_perp) / baseline_perp) * 100
                    f.write(f"  Pruning {ratio}: Perplexity={perp:.2f} ({degradation:+.1f}% change)\n")
            
            # Quantization
            for quant in ['INT8', 'INT4']:
                quant_data = model_data[(model_data['compression'] == 'quantization') & 
                                       (model_data['quant_level'] == quant)]
                if len(quant_data) > 0:
                    perp = quant_data['perplexity'].mean()
                    size = quant_data['model_size_mb'].mean()
                    degradation = ((perp - baseline_perp) / baseline_perp) * 100
                    size_reduction = ((baseline_size - size) / baseline_size) * 100
                    f.write(f"  {quant}: Perplexity={perp:.2f} ({degradation:+.1f}%), "
                           f"Size={size:.2f}MB ({size_reduction:.1f}% reduction)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS:\n")
        f.write("-" * 80 + "\n")
        f.write("1. INT4 quantization provides significant model size reduction (~75%)\n")
        f.write("2. Moderate pruning (10-50%) has minimal impact on perplexity\n")
        f.write("3. Extreme pruning (90%) can cause catastrophic performance degradation\n")
        f.write("4. Quantization alone is more stable than aggressive pruning\n")
        f.write("5. Combined approaches require careful balancing\n")
    
    print(f"✓ Saved analysis summary to {output_path}")
    print("\n" + "=" * 80)
    print("Analysis Summary:")
    print("=" * 80)
    with open(output_path, 'r') as f:
        print(f.read())

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("MICROSCALE LLM COMPRESSION - ENHANCED ANALYSIS")
    print("=" * 80 + "\n")
    
    # Setup paths
    csv_path = Path('metrics.csv')
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found!")
        return
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data(csv_path)
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    plot_perplexity_by_model_and_method(df, output_dir)
    plot_pruning_effect(df, output_dir)
    plot_quantization_effect(df, output_dir)
    plot_combined_effects(df, output_dir)
    plot_compression_tradeoffs(df, output_dir)
    plot_heatmaps_per_model(df, output_dir)
    
    # Generate summary
    print("\nGenerating summary report...")
    print("-" * 80)
    generate_summary_report(df, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  📊 {file.name}")
    for file in sorted(output_dir.glob('*.txt')):
        print(f"  📄 {file.name}")
    print()

if __name__ == '__main__':
    main()

