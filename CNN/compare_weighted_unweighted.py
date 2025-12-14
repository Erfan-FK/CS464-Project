#!/usr/bin/env python3
"""
Compare weighted vs unweighted loss training results.
Visualizes the impact of class weighting on imbalanced dataset performance.
"""

import argparse
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_class_names(class_names_file):
    """Load class names from file."""
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names


def parse_test_output(test_file):
    """Parse test output file to extract metrics and per-class data."""
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Extract overall metrics
    metrics = {}
    acc_match = re.search(r'Accuracy\s*:\s*([\d.]+)', content)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))
    
    macro_f1_match = re.search(r'Macro F1\s*:\s*([\d.]+)', content)
    if macro_f1_match:
        metrics['macro_f1'] = float(macro_f1_match.group(1))
    
    micro_f1_match = re.search(r'Micro F1\s*:\s*([\d.]+)', content)
    if micro_f1_match:
        metrics['micro_f1'] = float(micro_f1_match.group(1))
    
    weighted_f1_match = re.search(r'Weighted F1\s*:\s*([\d.]+)', content)
    if weighted_f1_match:
        metrics['weighted_f1'] = float(weighted_f1_match.group(1))
    
    # Extract per-class metrics
    per_class_data = []
    lines = content.split('\n')
    in_class_section = False
    
    for line in lines:
        if 'precision    recall  f1-score   support' in line:
            in_class_section = True
            continue
        
        if in_class_section:
            if line.strip() == '':
                continue
            if 'accuracy' in line.lower() or 'macro avg' in line or 'weighted avg' in line:
                break
            
            match = re.match(r'\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$', line)
            if match:
                class_name = match.group(1).strip()
                try:
                    precision = float(match.group(2))
                    recall = float(match.group(3))
                    f1 = float(match.group(4))
                    support = int(match.group(5))
                    per_class_data.append({
                        'class': class_name,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'support': support
                    })
                except ValueError:
                    continue
    
    return metrics, per_class_data


def analyze_class_imbalance(per_class_data):
    """Categorize classes by size."""
    df = pd.DataFrame(per_class_data)
    q1 = df['support'].quantile(0.25)
    q3 = df['support'].quantile(0.75)
    
    df['category'] = 'Medium'
    df.loc[df['support'] >= q3, 'category'] = 'Majority'
    df.loc[df['support'] <= q1, 'category'] = 'Minority'
    
    return df


def plot_training_comparison(df_weighted, df_unweighted, output_dir):
    """Compare training curves between weighted and unweighted."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training Loss
    ax = axes[0, 0]
    ax.plot(df_weighted['epoch'], df_weighted['train_loss'], 
           label='Weighted Loss', linewidth=2, alpha=0.8, color='#2ecc71')
    ax.plot(df_unweighted['epoch'], df_unweighted['train_loss'], 
           label='Unweighted Loss', linewidth=2, alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    ax.plot(df_weighted['epoch'], df_weighted['val_loss'], 
           label='Weighted Loss', linewidth=2, alpha=0.8, color='#2ecc71')
    ax.plot(df_unweighted['epoch'], df_unweighted['val_loss'], 
           label='Unweighted Loss', linewidth=2, alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[1, 0]
    ax.plot(df_weighted['epoch'], df_weighted['train_acc'], 
           label='Weighted Loss', linewidth=2, alpha=0.8, color='#2ecc71')
    ax.plot(df_unweighted['epoch'], df_unweighted['train_acc'], 
           label='Unweighted Loss', linewidth=2, alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Validation Accuracy
    ax = axes[1, 1]
    ax.plot(df_weighted['epoch'], df_weighted['val_acc'], 
           label='Weighted Loss', linewidth=2, alpha=0.8, color='#2ecc71')
    ax.plot(df_unweighted['epoch'], df_unweighted['val_acc'], 
           label='Unweighted Loss', linewidth=2, alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison_weighted_vs_unweighted.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: training_comparison_weighted_vs_unweighted.jpg")


def plot_test_metrics_comparison(metrics_weighted, metrics_unweighted, output_dir):
    """Compare overall test metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['Accuracy', 'Macro F1', 'Micro F1', 'Weighted F1']
    weighted_vals = [
        metrics_weighted.get('accuracy', 0),
        metrics_weighted.get('macro_f1', 0),
        metrics_weighted.get('micro_f1', 0),
        metrics_weighted.get('weighted_f1', 0)
    ]
    unweighted_vals = [
        metrics_unweighted.get('accuracy', 0),
        metrics_unweighted.get('macro_f1', 0),
        metrics_unweighted.get('micro_f1', 0),
        metrics_unweighted.get('weighted_f1', 0)
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, weighted_vals, width, label='Weighted Loss', 
                   alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x + width/2, unweighted_vals, width, label='Unweighted Loss', 
                   alpha=0.8, color='#e74c3c')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Performance: Weighted vs Unweighted Loss', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_metrics_comparison.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: test_metrics_comparison.jpg")


def plot_per_class_f1_comparison(per_class_weighted, per_class_unweighted, output_dir):
    """Compare per-class F1 scores."""
    df_w = pd.DataFrame(per_class_weighted)
    df_u = pd.DataFrame(per_class_unweighted)
    
    # Merge on class name
    df_merged = df_w.merge(df_u, on='class', suffixes=('_weighted', '_unweighted'))
    df_merged['f1_diff'] = df_merged['f1_weighted'] - df_merged['f1_unweighted']
    df_merged = df_merged.sort_values('f1_diff', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 16))
    
    y_pos = np.arange(len(df_merged))
    
    # Color based on which is better
    colors = ['#e74c3c' if diff < 0 else '#2ecc71' for diff in df_merged['f1_diff']]
    
    bars = ax.barh(y_pos, df_merged['f1_diff'], color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_merged['class'], fontsize=8)
    ax.set_xlabel('F1 Score Difference (Weighted - Unweighted)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Score: Impact of Weighted Loss', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Weighted Better'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Unweighted Better')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_f1_difference.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: per_class_f1_difference.jpg")


def plot_imbalance_impact(per_class_weighted, per_class_unweighted, output_dir):
    """Analyze impact on minority vs majority classes."""
    df_w = analyze_class_imbalance(per_class_weighted)
    df_u = analyze_class_imbalance(per_class_unweighted)
    
    # Merge
    df_merged = df_w.merge(df_u, on='class', suffixes=('_weighted', '_unweighted'))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['precision', 'recall', 'f1']
    categories = ['Minority', 'Medium', 'Majority']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        weighted_means = []
        unweighted_means = []
        
        for cat in categories:
            subset = df_merged[df_merged['category_weighted'] == cat]
            weighted_means.append(subset[f'{metric}_weighted'].mean())
            unweighted_means.append(subset[f'{metric}_unweighted'].mean())
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, weighted_means, width, label='Weighted Loss', 
                      alpha=0.8, color='#2ecc71')
        bars2 = ax.bar(x + width/2, unweighted_means, width, label='Unweighted Loss', 
                      alpha=0.8, color='#e74c3c')
        
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} by Class Size', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Impact of Weighted Loss on Minority vs Majority Classes', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'imbalance_impact_by_class_size.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: imbalance_impact_by_class_size.jpg")


def plot_minority_class_improvement(per_class_weighted, per_class_unweighted, output_dir):
    """Focus on minority class improvements."""
    df_w = analyze_class_imbalance(per_class_weighted)
    df_u = analyze_class_imbalance(per_class_unweighted)
    
    # Get minority classes
    minority_w = df_w[df_w['category'] == 'Minority']
    minority_u = df_u[df_u['category'] == 'Minority']
    
    # Merge
    df_merged = minority_w.merge(minority_u, on='class', suffixes=('_weighted', '_unweighted'))
    df_merged['f1_improvement'] = df_merged['f1_weighted'] - df_merged['f1_unweighted']
    df_merged = df_merged.sort_values('f1_improvement', ascending=True)
    
    if len(df_merged) == 0:
        print("⚠ No minority classes found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: F1 improvement for minority classes
    colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in df_merged['f1_improvement']]
    bars = ax1.barh(range(len(df_merged)), df_merged['f1_improvement'], 
                    color=colors, alpha=0.7)
    
    ax1.set_yticks(range(len(df_merged)))
    ax1.set_yticklabels(df_merged['class'], fontsize=9)
    ax1.set_xlabel('F1 Improvement (Weighted - Unweighted)', fontsize=12, fontweight='bold')
    ax1.set_title('Minority Classes: F1 Score Improvement', fontsize=13, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val, support) in enumerate(zip(bars, df_merged['f1_improvement'], 
                                                 df_merged['support_weighted'])):
        ax1.text(val + 0.005 if val >= 0 else val - 0.005, i, 
                f'{val:+.3f} (n={support})', 
                va='center', ha='left' if val >= 0 else 'right', fontsize=7)
    
    # Right: Scatter plot showing weighted vs unweighted F1
    ax2.scatter(df_merged['f1_unweighted'], df_merged['f1_weighted'], 
               s=100, alpha=0.6, color='#3498db')
    
    # Add diagonal line
    lims = [0, 1]
    ax2.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Equal Performance')
    
    ax2.set_xlabel('Unweighted F1 Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weighted F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Minority Classes: Weighted vs Unweighted F1', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Add class labels
    for _, row in df_merged.iterrows():
        ax2.annotate(row['class'], 
                    (row['f1_unweighted'], row['f1_weighted']),
                    fontsize=6, alpha=0.7, 
                    xytext=(3, 3), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'minority_class_improvement_analysis.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: minority_class_improvement_analysis.jpg")


def create_comparison_report(df_weighted, df_unweighted, 
                            metrics_weighted, metrics_unweighted,
                            per_class_weighted, per_class_unweighted, 
                            output_dir):
    """Create detailed comparison report."""
    report_path = output_dir / 'weighted_vs_unweighted_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WEIGHTED vs UNWEIGHTED LOSS COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Training comparison
        f.write("TRAINING PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Final Training Accuracy:\n")
        f.write(f"  Weighted:   {df_weighted['train_acc'].iloc[-1]:.4f}\n")
        f.write(f"  Unweighted: {df_unweighted['train_acc'].iloc[-1]:.4f}\n")
        f.write(f"  Difference: {df_weighted['train_acc'].iloc[-1] - df_unweighted['train_acc'].iloc[-1]:+.4f}\n\n")
        
        f.write(f"Final Validation Accuracy:\n")
        f.write(f"  Weighted:   {df_weighted['val_acc'].iloc[-1]:.4f}\n")
        f.write(f"  Unweighted: {df_unweighted['val_acc'].iloc[-1]:.4f}\n")
        f.write(f"  Difference: {df_weighted['val_acc'].iloc[-1] - df_unweighted['val_acc'].iloc[-1]:+.4f}\n\n")
        
        f.write(f"Best Validation Accuracy:\n")
        f.write(f"  Weighted:   {df_weighted['val_acc'].max():.4f}\n")
        f.write(f"  Unweighted: {df_unweighted['val_acc'].max():.4f}\n")
        f.write(f"  Difference: {df_weighted['val_acc'].max() - df_unweighted['val_acc'].max():+.4f}\n\n")
        
        # Test comparison
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:\n")
        f.write(f"  Weighted:   {metrics_weighted.get('accuracy', 0):.4f}\n")
        f.write(f"  Unweighted: {metrics_unweighted.get('accuracy', 0):.4f}\n")
        f.write(f"  Difference: {metrics_weighted.get('accuracy', 0) - metrics_unweighted.get('accuracy', 0):+.4f}\n\n")
        
        f.write(f"Macro F1 (average across all classes, unweighted):\n")
        f.write(f"  Weighted:   {metrics_weighted.get('macro_f1', 0):.4f}\n")
        f.write(f"  Unweighted: {metrics_unweighted.get('macro_f1', 0):.4f}\n")
        f.write(f"  Difference: {metrics_weighted.get('macro_f1', 0) - metrics_unweighted.get('macro_f1', 0):+.4f}\n\n")
        
        f.write(f"Weighted F1 (weighted by support):\n")
        f.write(f"  Weighted:   {metrics_weighted.get('weighted_f1', 0):.4f}\n")
        f.write(f"  Unweighted: {metrics_unweighted.get('weighted_f1', 0):.4f}\n")
        f.write(f"  Difference: {metrics_weighted.get('weighted_f1', 0) - metrics_unweighted.get('weighted_f1', 0):+.4f}\n\n")
        
        # Per-class analysis
        df_w = analyze_class_imbalance(per_class_weighted)
        df_u = analyze_class_imbalance(per_class_unweighted)
        df_merged = df_w.merge(df_u, on='class', suffixes=('_weighted', '_unweighted'))
        
        f.write("IMPACT ON CLASS CATEGORIES\n")
        f.write("-" * 80 + "\n")
        
        for category in ['Minority', 'Medium', 'Majority']:
            subset = df_merged[df_merged['category_weighted'] == category]
            if len(subset) > 0:
                f.write(f"\n{category.upper()} CLASSES (n={len(subset)}):\n")
                f.write(f"  Precision:\n")
                f.write(f"    Weighted:   {subset['precision_weighted'].mean():.4f}\n")
                f.write(f"    Unweighted: {subset['precision_unweighted'].mean():.4f}\n")
                f.write(f"    Difference: {subset['precision_weighted'].mean() - subset['precision_unweighted'].mean():+.4f}\n")
                
                f.write(f"  Recall:\n")
                f.write(f"    Weighted:   {subset['recall_weighted'].mean():.4f}\n")
                f.write(f"    Unweighted: {subset['recall_unweighted'].mean():.4f}\n")
                f.write(f"    Difference: {subset['recall_weighted'].mean() - subset['recall_unweighted'].mean():+.4f}\n")
                
                f.write(f"  F1 Score:\n")
                f.write(f"    Weighted:   {subset['f1_weighted'].mean():.4f}\n")
                f.write(f"    Unweighted: {subset['f1_unweighted'].mean():.4f}\n")
                f.write(f"    Difference: {subset['f1_weighted'].mean() - subset['f1_unweighted'].mean():+.4f}\n")
        
        # Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n")
        
        macro_diff = metrics_weighted.get('macro_f1', 0) - metrics_unweighted.get('macro_f1', 0)
        minority_subset = df_merged[df_merged['category_weighted'] == 'Minority']
        minority_f1_diff = minority_subset['f1_weighted'].mean() - minority_subset['f1_unweighted'].mean()
        
        f.write(f"\n1. Macro F1 (treats all classes equally): {macro_diff:+.4f}\n")
        if macro_diff > 0:
            f.write("   → Weighted loss IMPROVES average per-class performance\n")
        else:
            f.write("   → Unweighted loss performs better on average\n")
        
        f.write(f"\n2. Minority Class F1 Improvement: {minority_f1_diff:+.4f}\n")
        if minority_f1_diff > 0:
            f.write("   → Weighted loss HELPS minority classes\n")
        else:
            f.write("   → Weighted loss does not help minority classes\n")
        
        f.write(f"\n3. Overall Accuracy Difference: {metrics_weighted.get('accuracy', 0) - metrics_unweighted.get('accuracy', 0):+.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Saved: weighted_vs_unweighted_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Compare weighted vs unweighted training results")
    parser.add_argument("--weighted_results", type=str, required=True, 
                       help="Path to results.csv from weighted training")
    parser.add_argument("--unweighted_results", type=str, required=True, 
                       help="Path to results.csv from unweighted training")
    parser.add_argument("--weighted_test", type=str, required=True, 
                       help="Path to test.out from weighted training")
    parser.add_argument("--unweighted_test", type=str, required=True, 
                       help="Path to test.out from unweighted training")
    parser.add_argument("--class_names", type=str, required=True, 
                       help="Path to class_names.txt")
    parser.add_argument("--output_dir", type=str, default="comparison", 
                       help="Output directory for comparison plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("WEIGHTED vs UNWEIGHTED LOSS COMPARISON")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    df_weighted = pd.read_csv(args.weighted_results)
    df_unweighted = pd.read_csv(args.unweighted_results)
    class_names = load_class_names(args.class_names)
    
    metrics_weighted, per_class_weighted = parse_test_output(args.weighted_test)
    metrics_unweighted, per_class_unweighted = parse_test_output(args.unweighted_test)
    
    print(f"✓ Loaded weighted training: {len(df_weighted)} epochs")
    print(f"✓ Loaded unweighted training: {len(df_unweighted)} epochs")
    print(f"✓ Loaded {len(class_names)} classes")
    print(f"✓ Loaded {len(per_class_weighted)} per-class metrics (weighted)")
    print(f"✓ Loaded {len(per_class_unweighted)} per-class metrics (unweighted)\n")
    
    # Generate comparison plots
    print("Generating comparison visualizations...\n")
    
    plot_training_comparison(df_weighted, df_unweighted, output_dir)
    plot_test_metrics_comparison(metrics_weighted, metrics_unweighted, output_dir)
    
    if per_class_weighted and per_class_unweighted:
        plot_per_class_f1_comparison(per_class_weighted, per_class_unweighted, output_dir)
        plot_imbalance_impact(per_class_weighted, per_class_unweighted, output_dir)
        plot_minority_class_improvement(per_class_weighted, per_class_unweighted, output_dir)
    
    create_comparison_report(df_weighted, df_unweighted, 
                           metrics_weighted, metrics_unweighted,
                           per_class_weighted, per_class_unweighted, 
                           output_dir)
    
    print("\n" + "="*80)
    print(f"All comparison visualizations saved to: {output_dir.absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
