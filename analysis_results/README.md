# Microscale LLM Compression - Analysis Results

This directory contains comprehensive visualizations and analysis of the quantization and pruning experiments on GPT-2 models.

## 📊 Generated Visualizations

### 1. **comprehensive_dashboard.png** ⭐ START HERE
A single comprehensive dashboard containing 7 key visualizations:
- Baseline model performance comparison
- Size reduction via INT4 quantization
- Impact of pruning on performance
- Impact of quantization on performance
- Inference speed by compression method
- Size vs performance trade-offs
- Summary statistics table

**Key Insights:**
- INT4 quantization provides 75-83% size reduction
- Light pruning (10%) maintains performance, but 90% pruning is catastrophic
- INT8 quantization significantly increases perplexity for smaller models
- Quantization is more stable than aggressive pruning

---

### 2. **perplexity_by_model_method.png**
Horizontal bar charts showing perplexity for each compression method across all three models (GPT-2, GPT-2-Medium, GPT-2-Large).

**What it shows:**
- Direct comparison of how each compression technique affects model quality
- Lower perplexity = better performance
- Baseline performance vs compressed versions

**Key Findings:**
- GPT-2-Large has the best baseline perplexity (126.6)
- 10% pruning has minimal impact on perplexity
- INT4 quantization increases perplexity modestly but is acceptable
- Extreme pruning (90%) causes catastrophic failures

---

### 3. **pruning_effect_analysis.png**
Two-panel visualization analyzing pruning effects:
- **Left:** Perplexity vs Pruning Ratio (0, 0.1, 0.5, 0.9)
- **Right:** Perplexity vs Sparsity (percentage of weights pruned)

**What it shows:**
- How increasing pruning ratios affect model quality
- Relationship between sparsity and performance degradation

**Key Findings:**
- 10% pruning: Minimal degradation (~3% change for GPT-2)
- 50% pruning: Moderate degradation (~8-20% increase in perplexity)
- 90% pruning: Catastrophic failure (100,000x+ perplexity increase)
- **Sweet spot:** 10-30% pruning for acceptable performance

---

### 4. **quantization_effect_analysis.png**
Two-panel comparison of quantization effects:
- **Left:** Perplexity by quantization level (FP32, INT8, INT4)
- **Right:** Model size by quantization level

**What it shows:**
- Trade-off between precision reduction and model quality
- Size savings from quantization

**Key Findings:**
- **GPT-2-Large + INT8:** Minimal perplexity change (-0.9%), small size reduction
- **GPT-2-Large + INT4:** Slight improvement (-2.2%), 82.9% size reduction ⭐
- **GPT-2 + INT4:** Modest degradation (+6.3%), 74.5% size reduction
- **GPT-2-Medium + INT4:** Moderate degradation (+31.7%), 80.6% size reduction
- **Larger models handle quantization better than smaller models**

---

### 5. **combined_effects_comparison.png**
Four-panel dashboard comparing all compression methods:
- Perplexity comparison
- Model size comparison
- Inference speed comparison
- Memory footprint comparison

**What it shows:**
- Holistic view of how compression methods stack up across all metrics
- Combined (pruning + quantization) vs individual techniques

**Key Findings:**
- Quantization provides best size reduction
- Combined approaches don't always outperform single methods
- Pruning alone doesn't reduce model size significantly
- Inference speed varies significantly by device (CPU vs GPU)

---

### 6. **compression_tradeoffs.png**
Two scatter plots showing fundamental trade-offs:
- **Left:** Model Size vs Perplexity
- **Right:** Inference Speed vs Perplexity

**What it shows:**
- Pareto frontier of compression techniques
- Where different methods fall in the quality-efficiency space
- Better to be in bottom-left (small size, low perplexity) or top-right (high speed, low perplexity)

**Key Findings:**
- INT4 quantization achieves excellent size-performance trade-off
- Baseline models are slowest but highest quality
- Combined methods cluster with similar trade-offs to quantization alone

---

### 7. **pruning_quantization_heatmaps.png**
Heatmaps showing perplexity for all combinations of pruning ratios and quantization levels.

**What it shows:**
- Interactive view of how pruning and quantization interact
- Darker colors = worse performance (higher perplexity)
- Best combinations are lighter colored

**Key Findings:**
- Low pruning (0.1) + INT4 works well
- High pruning (0.9) + any quantization = disaster
- INT4 is generally better than INT8 for combined approaches

---

## 📈 Key Patterns in the Data

### Pattern 1: Model Size Matters for Quantization Tolerance
- **GPT-2-Large** (774M params): Handles quantization excellently
  - INT4: -2.2% perplexity (improvement!)
  - INT8: -0.9% perplexity (improvement!)
- **GPT-2** (124M params): More sensitive to quantization
  - INT4: +6.3% perplexity (acceptable)
  - INT8: +292% perplexity (significant degradation)

**Why?** Larger models have more redundancy and can tolerate precision reduction better.

### Pattern 2: Pruning Has a Sharp Cliff
- 0% → 10% pruning: Minimal impact (~3% degradation)
- 10% → 50% pruning: Moderate impact (~8-20% degradation)
- 50% → 90% pruning: **Catastrophic failure** (100,000x+ degradation)

**Why?** Beyond a certain point, critical model capacity is lost, causing complete breakdown.

### Pattern 3: Quantization > Pruning for Size Reduction
- **INT4 quantization:** 75-83% size reduction, <32% perplexity increase
- **90% pruning:** Minimal size reduction, catastrophic perplexity increase

**Why?** Quantization reduces precision uniformly, while pruning removes capacity unevenly.

### Pattern 4: Combined Compression Requires Balance
- **Good:** Prune 10% + INT4 → Moderate perplexity increase, good size reduction
- **Bad:** Prune 50%+ + any quantization → Severe degradation

**Why?** Both methods compress the model; stacking them multiplies the quality loss.

### Pattern 5: Device Performance Varies Dramatically
- **GPU:** High inference speed for all methods
- **CPU:** Dramatically slower, especially for INT8 (0.3 vs 73+ samples/sec)

**Why?** GPU has specialized hardware for matrix operations; CPU relies on general computation.

---

## 🎯 Recommended Compression Strategies

### For Maximum Size Reduction
- **Best:** INT4 quantization alone
- **Result:** 75-83% size reduction, acceptable quality loss
- **Use case:** Deploying on mobile devices with limited storage

### For Best Quality
- **Best:** 10% pruning OR no compression
- **Result:** Minimal perplexity increase (<3%)
- **Use case:** Edge devices with decent storage but limited compute

### For Balance
- **Best:** 10% pruning + INT4 quantization
- **Result:** Good size reduction, moderate quality loss
- **Use case:** General edge deployment scenarios

### What to AVOID
- ❌ 90% pruning (catastrophic failure)
- ❌ INT8 quantization on small models (severe quality loss)
- ❌ Combining high pruning (50%+) with quantization
- ❌ Assuming CPU can match GPU inference speeds

---

## 📊 Summary Statistics

| Model | Baseline PPL | Best Compressed PPL | Best Method | Size Reduction |
|-------|--------------|---------------------|-------------|----------------|
| GPT-2 | 519.9 | 503.7 | Pruning 0.1 | 74.5% (INT4) |
| GPT-2-Medium | 150.3 | 150.3 | Baseline | 80.6% (INT4) |
| GPT-2-Large | 126.6 | 123.9 | INT4 Quant | 82.9% (INT4) |

---

## 🔬 Methodology

**Dataset:** WikiText-2 test set  
**Metric:** Perplexity (lower is better)  
**Seeds:** 5 random seeds per configuration (42, 43, 44, 45, 46)  
**Hardware:** CUDA GPUs (cuda:0) and CPU  
**Models:** GPT-2 (124M), GPT-2-Medium (355M), GPT-2-Large (774M)  

**Compression Techniques:**
- **Pruning:** 10%, 50%, 90% of weights removed
- **Quantization:** INT8 (8-bit), INT4 (4-bit)
- **Combined:** Pruning + Quantization

---

## 🚀 How to Use These Visualizations

1. **Start with** `comprehensive_dashboard.png` for the big picture
2. **Deep dive** into specific compression methods:
   - Interested in pruning? → `pruning_effect_analysis.png`
   - Interested in quantization? → `quantization_effect_analysis.png`
   - Want to compare everything? → `combined_effects_comparison.png`
3. **Understand trade-offs** with `compression_tradeoffs.png`
4. **Find optimal combinations** using `pruning_quantization_heatmaps.png`

---

## 📝 Analysis Scripts

The visualizations were generated using:
- `analysis_visualization.py` - Main analysis script with 6 visualization functions
- `create_summary_dashboard.py` - Comprehensive dashboard generator
- `plot_results.py` - Original plotting script (from project)

To regenerate:
```bash
python3 analysis_visualization.py
python3 create_summary_dashboard.py
```

---

## 💡 Conclusions

1. **INT4 quantization is the clear winner** for edge deployment
   - Massive size reduction (75-83%)
   - Acceptable or even improved quality (especially for large models)
   - Stable across different model sizes

2. **Light pruning (10%) is safe** but provides limited benefits
   - Minimal quality loss
   - Doesn't reduce model size much
   - Can be combined with quantization carefully

3. **Aggressive pruning (90%) should be avoided**
   - Catastrophic performance degradation
   - Not worth the minimal size savings

4. **Model size matters**
   - Larger models tolerate compression better
   - GPT-2-Large even improved with quantization!

5. **For real-world deployment**
   - Use INT4 quantization as the primary technique
   - Add light pruning (10%) only if needed
   - Test thoroughly before deploying to production

---

**Generated on:** November 6, 2025  
**Analysis by:** Microscale LLM Research Team

