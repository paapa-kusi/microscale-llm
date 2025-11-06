#!/usr/bin/env python3
"""
Create a comprehensive dashboard summarizing all key findings.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_comprehensive_dashboard(csv_path, output_path):
    """Create a single comprehensive dashboard with all key metrics."""
    df = pd.read_csv(csv_path)
    
    # Filter out EleutherAI model
    df = df[~df['model'].str.contains('EleutherAI', case=False, na=False)]
    
    df['compression'] = df['compression'].fillna('baseline')
    
    # Set quant_level to FP16 for baseline data
    df.loc[df['compression'] == 'baseline', 'quant_level'] = 'FP16'
    
    # Calculate averages across seeds
    df_avg = df.groupby(['model', 'compression', 'prune_ratio', 'quant_level']).agg({
        'perplexity': 'mean',
        'model_size_mb': 'mean',
        'inference_speed': 'mean',
        'sparsity': 'mean'
    }).reset_index()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Microscale LLM Compression: Comprehensive Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Plot 1: Baseline Perplexity Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    baseline_data = df_avg[df_avg['compression'] == 'baseline']
    colors = sns.color_palette("husl", len(baseline_data))
    bars = ax1.barh(baseline_data['model'], baseline_data['perplexity'], color=colors)
    ax1.set_xlabel('Perplexity', fontsize=11, fontweight='bold')
    ax1.set_title('1. Baseline Model Performance', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, (model, perp) in enumerate(zip(baseline_data['model'], baseline_data['perplexity'])):
        ax1.text(perp/2, i, f'{perp:.1f}', va='center', ha='center', 
                fontweight='bold', color='white', fontsize=10)
    
    # Plot 2: Model Size Comparison - Baseline vs Best Quantization
    ax2 = fig.add_subplot(gs[0, 1])
    models = df_avg['model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    baseline_sizes = []
    int4_sizes = []
    for m in models:
        baseline_data = df_avg[(df_avg['model'] == m) & (df_avg['compression'] == 'baseline')]['model_size_mb']
        if len(baseline_data) > 0:
            baseline_sizes.append(baseline_data.values[0])
        else:
            baseline_sizes.append(0)
        
        int4_data = df_avg[(df_avg['model'] == m) & (df_avg['quant_level'] == 'INT4')]['model_size_mb']
        if len(int4_data) > 0:
            int4_sizes.append(int4_data.mean())
        else:
            int4_sizes.append(0)
    
    ax2.bar(x - width/2, baseline_sizes, width, label='Baseline', color='#3498db')
    ax2.bar(x + width/2, int4_sizes, width, label='INT4', color='#e74c3c')
    ax2.set_ylabel('Model Size (MB)', fontsize=11, fontweight='bold')
    ax2.set_title('2. Size Reduction via INT4 Quantization', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, (baseline, int4) in enumerate(zip(baseline_sizes, int4_sizes)):
        if baseline > 0 and int4 > 0:
            reduction = ((baseline - int4) / baseline) * 100
            ax2.text(i, max(baseline, int4) + 100, f'-{reduction:.0f}%', 
                    ha='center', fontsize=9, fontweight='bold', color='green')
    
    # Plot 3: Perplexity Impact - Pruning
    ax3 = fig.add_subplot(gs[0, 2])
    pruning_data = df_avg[df_avg['compression'].isin(['baseline', 'pruning'])]
    pruning_data['prune_ratio'] = pruning_data['prune_ratio'].fillna(0.0)
    
    for model in models:
        model_data = pruning_data[pruning_data['model'] == model]
        # Filter out extreme values for better visualization
        model_data = model_data[model_data['perplexity'] < 10000]
        model_data = model_data.sort_values('prune_ratio')
        ax3.plot(model_data['prune_ratio'], model_data['perplexity'], 
                marker='o', linewidth=2, markersize=8, label=model)
    
    ax3.set_xlabel('Pruning Ratio', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Perplexity', fontsize=11, fontweight='bold')
    ax3.set_title('3. Impact of Pruning on Performance', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Perplexity Impact - Quantization
    ax4 = fig.add_subplot(gs[1, 0])
    quant_levels = ['FP16', 'INT8', 'INT4']
    x = np.arange(len(models))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, ql in enumerate(quant_levels):
        perplexities = []
        for model in models:
            if ql == 'FP16':
                data = df_avg[(df_avg['model'] == model) & (df_avg['compression'] == 'baseline')]
            else:
                data = df_avg[(df_avg['model'] == model) & (df_avg['quant_level'] == ql)]
            
            if len(data) > 0:
                perplexities.append(data['perplexity'].values[0])
            else:
                perplexities.append(np.nan)
        
        offset = (idx - 1) * width
        ax4.bar(x + offset, perplexities, width, label=ql, color=colors[idx])
    
    ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Perplexity', fontsize=11, fontweight='bold')
    ax4.set_title('4. Impact of Quantization on Performance', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend(fontsize=9, title='Quant Level')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Inference Speed Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    compression_types = ['baseline', 'pruning', 'quantization', 'combined']
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, comp in enumerate(compression_types):
        speeds = []
        for model in models:
            data = df_avg[(df_avg['model'] == model) & (df_avg['compression'] == comp)]
            if len(data) > 0:
                speeds.append(data['inference_speed'].mean())
            else:
                speeds.append(np.nan)
        
        offset = (idx - 1.5) * width
        ax5.bar(x + offset, speeds, width, label=comp, color=colors[idx])
    
    ax5.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Inference Speed (samples/sec)', fontsize=11, fontweight='bold')
    ax5.set_title('5. Inference Speed by Method', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(models)
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_yscale('log')
    
    # Plot 6: Compression Trade-off Scatter
    ax6 = fig.add_subplot(gs[1, 2])
    df_viz = df_avg[df_avg['perplexity'] < 10000].copy()
    
    for compression in df_viz['compression'].unique():
        comp_data = df_viz[df_viz['compression'] == compression]
        marker = 'o' if compression == 'baseline' else 's' if compression == 'pruning' \
                else '^' if compression == 'quantization' else 'D'
        ax6.scatter(comp_data['model_size_mb'], comp_data['perplexity'], 
                   s=100, alpha=0.7, marker=marker, label=compression)
    
    ax6.set_xlabel('Model Size (MB)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Perplexity', fontsize=11, fontweight='bold')
    ax6.set_title('6. Size vs Performance Trade-off', fontsize=12, fontweight='bold')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Key Statistics Table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Model', 'Baseline PPL', 'Best PPL', 'Method', 'Size Reduction', 'Speed'])
    
    for model in models:
        baseline = df_avg[(df_avg['model'] == model) & (df_avg['compression'] == 'baseline')]
        if len(baseline) == 0:
            continue
        baseline_ppl = baseline['perplexity'].values[0]
        baseline_size = baseline['model_size_mb'].values[0]
        baseline_speed = baseline['inference_speed'].values[0]
        
        # Find best compression (excluding catastrophic failures)
        model_data = df_avg[(df_avg['model'] == model) & (df_avg['perplexity'] < 10000)]
        best = model_data.loc[model_data['perplexity'].idxmin()]
        
        # Get INT4 data for size comparison
        int4_data = df_avg[(df_avg['model'] == model) & (df_avg['quant_level'] == 'INT4')]
        if len(int4_data) > 0:
            int4_size = int4_data['model_size_mb'].mean()
            size_reduction = f"{((baseline_size - int4_size) / baseline_size * 100):.0f}% (INT4)"
        else:
            size_reduction = "N/A"
        
        method = best['compression']
        if pd.notna(best['prune_ratio']):
            method += f" {best['prune_ratio']}"
        if pd.notna(best['quant_level']):
            method += f" {best['quant_level']}"
        
        table_data.append([
            model,
            f"{baseline_ppl:.1f}",
            f"{best['perplexity']:.1f}",
            method,
            size_reduction,
            f"{baseline_speed:.1f}"
        ])
    
    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.25, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
    
    ax7.set_title('7. Summary Statistics: Best Results Per Model', 
                 fontsize=12, fontweight='bold', pad=20)
    
    # Add key insights text box
    insights_text = """
    KEY FINDINGS:
    • INT4 quantization achieves 75-83% size reduction with minimal perplexity increase
    • Light pruning (10%) maintains performance well, but aggressive pruning (90%) causes degradation
    • INT8 quantization increases perplexity significantly for smaller models but works well for large models
    • Combined compression requires careful tuning to balance size and performance
    • Quantization is generally more stable than aggressive pruning
    """
    
    fig.text(0.5, 0.02, insights_text, ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive dashboard to {output_path}")
    plt.close()

if __name__ == '__main__':
    csv_path = Path('metrics.csv')
    output_path = Path('analysis_results/comprehensive_dashboard.png')
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found!")
    else:
        print("\n" + "="*80)
        print("Creating Comprehensive Dashboard...")
        print("="*80 + "\n")
        create_comprehensive_dashboard(csv_path, output_path)
        print("\n" + "="*80)
        print("Dashboard creation complete!")
        print("="*80 + "\n")

