#!/usr/bin/env python3
"""
Comprehensive visualization script for CNN training and test results.
Generates multiple plots and saves them to the 'visualize' folder.
"""

import argparse
import sys
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch import nn

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_class_names(class_names_file):
    """Load class names from file."""
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names


def parse_training_log(log_file):
    """Parse training log file to extract epoch, train/val loss and accuracy."""
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for epoch line
        epoch_match = re.match(r'Epoch (\d+)/\d+', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            
            # Next line should be train metrics
            if i + 1 < len(lines):
                train_line = lines[i + 1].strip()
                train_match = re.search(r'Train loss: ([\d.]+), acc: ([\d.]+)', train_line)
                if train_match:
                    train_loss = float(train_match.group(1))
                    train_acc = float(train_match.group(2))
                    
                    # Next line should be val metrics
                    if i + 2 < len(lines):
                        val_line = lines[i + 2].strip()
                        val_match = re.search(r'Val\s+loss: ([\d.]+), acc: ([\d.]+)', val_line)
                        if val_match:
                            val_loss = float(val_match.group(1))
                            val_acc = float(val_match.group(2))
                            
                            epochs.append(epoch)
                            train_losses.append(train_loss)
                            train_accs.append(train_acc)
                            val_losses.append(val_loss)
                            val_accs.append(val_acc)
        i += 1
    
    return epochs, train_losses, train_accs, val_losses, val_accs


def parse_test_output(test_file):
    """Parse test output file to extract metrics and confusion matrix."""
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
    
    # Extract per-class metrics
    per_class_data = []
    lines = content.split('\n')
    in_class_section = False
    
    for line in lines:
        if 'precision    recall  f1-score   support' in line:
            in_class_section = True
            continue
        
        if in_class_section:
            # Stop at empty line or summary lines
            if line.strip() == '':
                continue
            if 'accuracy' in line.lower() or 'macro avg' in line or 'weighted avg' in line:
                break
            
            # Parse class line - use regex to extract the numeric values at the end
            # Format: "     ClassName     0.1234    0.5678    0.9012        123"
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


def plot_training_loss(df, output_dir):
    """Plot training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2, alpha=0.8)
    ax.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_validation_loss.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: training_validation_loss.jpg")


def plot_training_accuracy(df, output_dir):
    """Plot training and validation accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['train_acc'], label='Training Accuracy', linewidth=2, alpha=0.8)
    ax.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_validation_accuracy.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: training_validation_accuracy.jpg")


def plot_combined_metrics(df, output_dir):
    """Plot combined loss and accuracy in subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss subplot
    ax1.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Over Epochs', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2.plot(df['epoch'], df['train_acc'], label='Training Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Over Epochs', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_training_metrics.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: combined_training_metrics.jpg")


def plot_per_class_f1(per_class_data, output_dir):
    """Plot per-class F1 scores as a bar chart."""
    df = pd.DataFrame(per_class_data)
    df = df.sort_values('f1', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(range(len(df)), df['f1'], color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['class'], fontsize=8)
    ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Scores (Test Set)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['f1'])):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_f1_scores.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: per_class_f1_scores.jpg")


def plot_confusion_matrix_from_per_class(per_class_data, class_names, output_dir):
    """Generate confusion matrix visualization from per-class data.
    
    Note: Since we can't parse the truncated confusion matrix from test output,
    we create a simplified visualization showing diagonal (correct predictions)
    and use the per-class recall to estimate the confusion matrix structure.
    """
    if not per_class_data:
        print("⚠ Warning: No per-class data available, skipping confusion matrix plots")
        return
    
    # Create a confusion matrix estimate from per-class metrics
    # This is an approximation - diagonal shows correct predictions based on recall
    n_classes = len(class_names)
    cm_estimate = np.zeros((n_classes, n_classes))
    
    # Map class names to indices
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for item in per_class_data:
        class_name = item['class']
        if class_name in class_to_idx:
            idx = class_to_idx[class_name]
            support = item['support']
            recall = item['recall']
            
            # Diagonal: correct predictions
            cm_estimate[idx, idx] = support * recall
            
            # Off-diagonal: distribute errors uniformly (approximation)
            errors = support * (1 - recall)
            if errors > 0 and n_classes > 1:
                error_per_class = errors / (n_classes - 1)
                for j in range(n_classes):
                    if j != idx:
                        cm_estimate[idx, j] = error_per_class
    
    # Plot estimated confusion matrix
    fig, ax = plt.subplots(figsize=(20, 18))
    
    sns.heatmap(cm_estimate, annot=False, fmt='.0f', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count (Estimated)'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Estimated from Per-Class Metrics (Test Set)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=90, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_estimated.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: confusion_matrix_estimated.jpg")
    
    # Plot normalized confusion matrix (by row)
    cm_normalized = cm_estimate / cm_estimate.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    fig, ax = plt.subplots(figsize=(20, 18))
    
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='RdYlGn', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Proportion'}, ax=ax, vmin=0, vmax=1)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Confusion Matrix - Estimated (Test Set)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=90, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized_estimated.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: confusion_matrix_normalized_estimated.jpg")
    
    # Create a diagonal-focused view showing per-class accuracy
    fig, ax = plt.subplots(figsize=(12, 14))
    
    recalls = []
    class_labels = []
    for item in per_class_data:
        recalls.append(item['recall'])
        class_labels.append(item['class'])
    
    colors = plt.cm.RdYlGn(np.array(recalls))
    bars = ax.barh(range(len(recalls)), recalls, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(recalls)))
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_xlabel('Recall (Correct Predictions / Total)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Recall (Diagonal of Confusion Matrix)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, recalls)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_recall_diagonal.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: per_class_recall_diagonal.jpg")


def plot_class_performance_comparison(per_class_data, output_dir):
    """Plot comparison of precision, recall, and F1 for each class."""
    df = pd.DataFrame(per_class_data)
    df = df.sort_values('f1', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax.barh(x - width, df['precision'], width, label='Precision', alpha=0.8)
    bars2 = ax.barh(x, df['recall'], width, label='Recall', alpha=0.8)
    bars3 = ax.barh(x + width, df['f1'], width, label='F1 Score', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(df['class'], fontsize=8)
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics (Test Set)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_performance_comparison.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: class_performance_comparison.jpg")


def plot_learning_rate_schedule(df, output_dir):
    """Plot learning rate schedule over epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['lr'], linewidth=2, alpha=0.8, color='red')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_schedule.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: learning_rate_schedule.jpg")


def plot_top_bottom_classes(per_class_data, output_dir, n=10):
    """Plot top and bottom performing classes by F1 score."""
    df = pd.DataFrame(per_class_data)
    df_sorted = df.sort_values('f1', ascending=False)
    
    top_n = df_sorted.head(n)
    bottom_n = df_sorted.tail(n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top performers
    colors_top = plt.cm.Greens(np.linspace(0.5, 0.9, len(top_n)))
    bars1 = ax1.barh(range(len(top_n)), top_n['f1'], color=colors_top, alpha=0.8)
    ax1.set_yticks(range(len(top_n)))
    ax1.set_yticklabels(top_n['class'], fontsize=9)
    ax1.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {n} Classes by F1 Score', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars1, top_n['f1'])):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
    
    # Bottom performers
    colors_bottom = plt.cm.Reds(np.linspace(0.5, 0.9, len(bottom_n)))
    bars2 = ax2.barh(range(len(bottom_n)), bottom_n['f1'], color=colors_bottom, alpha=0.8)
    ax2.set_yticks(range(len(bottom_n)))
    ax2.set_yticklabels(bottom_n['class'], fontsize=9)
    ax2.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
    ax2.set_title(f'Bottom {n} Classes by F1 Score', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars2, bottom_n['f1'])):
        ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_bottom_classes.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: top_bottom_classes.jpg")


def plot_class_support_distribution(per_class_data, output_dir):
    """Plot distribution of samples per class in test set."""
    df = pd.DataFrame(per_class_data)
    df = df.sort_values('support', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(range(len(df)), df['support'], color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['class'], fontsize=8)
    ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Sample Distribution per Class', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['support'])):
        ax.text(val + 1, i, f'{val}', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_support_distribution.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: class_support_distribution.jpg")


def plot_training_phases(df, output_dir):
    """Identify and visualize different training phases."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot validation accuracy with different colors for different LR phases
    unique_lrs = df['lr'].unique()
    
    for lr in unique_lrs:
        mask = df['lr'] == lr
        subset = df[mask]
        ax.plot(subset['epoch'], subset['val_acc'], 
               label=f'LR = {lr:.0e}', linewidth=2, alpha=0.8, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training Phases by Learning Rate', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_phases.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: training_phases.jpg")


def analyze_class_imbalance(per_class_data):
    """Analyze class imbalance and categorize classes into majority/minority."""
    df = pd.DataFrame(per_class_data)
    
    # Calculate statistics
    total_samples = df['support'].sum()
    df['percentage'] = (df['support'] / total_samples) * 100
    
    # Define thresholds for majority/minority classes
    median_support = df['support'].median()
    q1_support = df['support'].quantile(0.25)
    q3_support = df['support'].quantile(0.75)
    
    # Categorize classes
    df['category'] = 'Medium'
    df.loc[df['support'] >= q3_support, 'category'] = 'Majority'
    df.loc[df['support'] <= q1_support, 'category'] = 'Minority'
    
    return df


def plot_class_imbalance_analysis(per_class_data, output_dir):
    """Plot comprehensive class imbalance analysis."""
    df = analyze_class_imbalance(per_class_data)
    
    # Sort by support
    df_sorted = df.sort_values('support', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Color by category
    colors = {'Minority': '#e74c3c', 'Medium': '#f39c12', 'Majority': '#27ae60'}
    bar_colors = [colors[cat] for cat in df_sorted['category']]
    
    bars = ax.barh(range(len(df_sorted)), df_sorted['support'], color=bar_colors, alpha=0.8)
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['class'], fontsize=8)
    ax.set_xlabel('Number of Test Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Imbalance Distribution (Test Set)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val, pct) in enumerate(zip(bars, df_sorted['support'], df_sorted['percentage'])):
        ax.text(val + 1, i, f'{val} ({pct:.1f}%)', va='center', fontsize=6)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['Majority'], alpha=0.8, label='Majority (Q3+)'),
        Patch(facecolor=colors['Medium'], alpha=0.8, label='Medium (Q1-Q3)'),
        Patch(facecolor=colors['Minority'], alpha=0.8, label='Minority (Q1-)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_imbalance_distribution.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: class_imbalance_distribution.jpg")


def plot_performance_by_class_size(per_class_data, output_dir):
    """Plot how performance metrics vary with class size."""
    df = analyze_class_imbalance(per_class_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics_to_plot = ['precision', 'recall', 'f1']
    colors = {'Minority': '#e74c3c', 'Medium': '#f39c12', 'Majority': '#27ae60'}
    
    # Scatter plots for each metric
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        for category in ['Minority', 'Medium', 'Majority']:
            subset = df[df['category'] == category]
            ax.scatter(subset['support'], subset[metric], 
                      label=category, alpha=0.7, s=80, color=colors[category])
        
        # Add trend line
        z = np.polyfit(df['support'], df[metric], 1)
        p = np.poly1d(z)
        ax.plot(df['support'].sort_values(), p(df['support'].sort_values()), 
               "r--", alpha=0.5, linewidth=2, label='Trend')
        
        ax.set_xlabel('Number of Samples', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} vs Class Size', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    # Summary statistics in 4th subplot
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "CLASS IMBALANCE IMPACT SUMMARY\n\n"
    
    for category in ['Minority', 'Medium', 'Majority']:
        subset = df[df['category'] == category]
        summary_text += f"{category} Classes (n={len(subset)}):\n"
        summary_text += f"  Avg Support: {subset['support'].mean():.1f}\n"
        summary_text += f"  Avg Precision: {subset['precision'].mean():.3f}\n"
        summary_text += f"  Avg Recall: {subset['recall'].mean():.3f}\n"
        summary_text += f"  Avg F1: {subset['f1'].mean():.3f}\n\n"
    
    # Calculate correlation
    corr_f1 = df['support'].corr(df['f1'])
    summary_text += f"Correlation (Support vs F1): {corr_f1:.3f}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_class_size.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: performance_vs_class_size.jpg")


def plot_majority_minority_comparison(per_class_data, output_dir):
    """Compare majority vs minority class performance."""
    df = analyze_class_imbalance(per_class_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['precision', 'recall', 'f1']
    categories = ['Minority', 'Medium', 'Majority']
    colors_map = {'Minority': '#e74c3c', 'Medium': '#f39c12', 'Majority': '#27ae60'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        data_by_category = [df[df['category'] == cat][metric].values for cat in categories]
        
        bp = ax.boxplot(data_by_category, labels=categories, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        for patch, category in zip(bp['boxes'], categories):
            patch.set_facecolor(colors_map[category])
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} Distribution by Class Size', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        # Add mean values as text
        for i, category in enumerate(categories):
            mean_val = df[df['category'] == category][metric].mean()
            ax.text(i + 1, 0.05, f'μ={mean_val:.3f}', 
                   ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'majority_minority_comparison.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: majority_minority_comparison.jpg")


def plot_minority_class_focus(per_class_data, output_dir):
    """Detailed view of minority class performance."""
    df = analyze_class_imbalance(per_class_data)
    minority_df = df[df['category'] == 'Minority'].sort_values('f1', ascending=True)
    
    if len(minority_df) == 0:
        print("⚠ No minority classes found, skipping minority class focus plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: F1 scores for minority classes
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(minority_df)))
    bars = ax1.barh(range(len(minority_df)), minority_df['f1'], color=colors, alpha=0.8)
    
    ax1.set_yticks(range(len(minority_df)))
    ax1.set_yticklabels(minority_df['class'], fontsize=9)
    ax1.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Minority Class Performance (F1 Score)', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val, support) in enumerate(zip(bars, minority_df['f1'], minority_df['support'])):
        ax1.text(val + 0.02, i, f'{val:.3f} (n={support})', va='center', fontsize=8)
    
    # Right: Precision-Recall comparison for minority classes
    x = np.arange(len(minority_df))
    width = 0.35
    
    bars1 = ax2.barh(x - width/2, minority_df['precision'], width, 
                     label='Precision', alpha=0.8, color='#3498db')
    bars2 = ax2.barh(x + width/2, minority_df['recall'], width, 
                     label='Recall', alpha=0.8, color='#e74c3c')
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(minority_df['class'], fontsize=9)
    ax2.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Minority Class: Precision vs Recall', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'minority_class_detailed_analysis.jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: minority_class_detailed_analysis.jpg")


def create_summary_report(df, metrics, per_class_data, output_dir):
    """Create a text summary report."""
    report_path = output_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CNN TRAINING AND TEST RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Training summary
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Epochs: {df['epoch'].max()}\n")
        f.write(f"Final Training Loss: {df['train_loss'].iloc[-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {df['train_acc'].iloc[-1]:.4f}\n")
        f.write(f"Final Validation Loss: {df['val_loss'].iloc[-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {df['val_acc'].iloc[-1]:.4f}\n")
        f.write(f"Best Validation Accuracy: {df['val_acc'].max():.4f} (Epoch {df.loc[df['val_acc'].idxmax(), 'epoch']:.0f})\n")
        f.write(f"Best Training Accuracy: {df['train_acc'].max():.4f} (Epoch {df.loc[df['train_acc'].idxmax(), 'epoch']:.0f})\n")
        f.write(f"Lowest Validation Loss: {df['val_loss'].min():.4f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']:.0f})\n\n")
        
        # Test summary
        f.write("TEST SET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"Test Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}\n\n")
        
        # Per-class summary
        df_classes = pd.DataFrame(per_class_data)
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean F1 Score: {df_classes['f1'].mean():.4f}\n")
        f.write(f"Std F1 Score: {df_classes['f1'].std():.4f}\n")
        f.write(f"Best Class: {df_classes.loc[df_classes['f1'].idxmax(), 'class']} (F1: {df_classes['f1'].max():.4f})\n")
        f.write(f"Worst Class: {df_classes.loc[df_classes['f1'].idxmin(), 'class']} (F1: {df_classes['f1'].min():.4f})\n\n")
        
        # Class imbalance analysis
        if per_class_data:
            df_imbalance = analyze_class_imbalance(per_class_data)
            
            f.write("CLASS IMBALANCE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Test Samples: {df_imbalance['support'].sum()}\n")
            f.write(f"Median Class Size: {df_imbalance['support'].median():.0f}\n")
            f.write(f"Min Class Size: {df_imbalance['support'].min():.0f}\n")
            f.write(f"Max Class Size: {df_imbalance['support'].max():.0f}\n")
            f.write(f"Imbalance Ratio (Max/Min): {df_imbalance['support'].max() / df_imbalance['support'].min():.2f}\n\n")
            
            for category in ['Minority', 'Medium', 'Majority']:
                subset = df_imbalance[df_imbalance['category'] == category]
                if len(subset) > 0:
                    f.write(f"{category.upper()} CLASSES (n={len(subset)}):\n")
                    f.write(f"  Support Range: {subset['support'].min():.0f} - {subset['support'].max():.0f}\n")
                    f.write(f"  Avg Support: {subset['support'].mean():.1f}\n")
                    f.write(f"  Avg Precision: {subset['precision'].mean():.4f} (±{subset['precision'].std():.4f})\n")
                    f.write(f"  Avg Recall: {subset['recall'].mean():.4f} (±{subset['recall'].std():.4f})\n")
                    f.write(f"  Avg F1: {subset['f1'].mean():.4f} (±{subset['f1'].std():.4f})\n\n")
            
            # Correlation analysis
            corr_prec = df_imbalance['support'].corr(df_imbalance['precision'])
            corr_rec = df_imbalance['support'].corr(df_imbalance['recall'])
            corr_f1 = df_imbalance['support'].corr(df_imbalance['f1'])
            
            f.write("CORRELATION (Support vs Performance):\n")
            f.write(f"  Support vs Precision: {corr_prec:+.4f}\n")
            f.write(f"  Support vs Recall: {corr_rec:+.4f}\n")
            f.write(f"  Support vs F1: {corr_f1:+.4f}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved: summary_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Visualize CNN training and test results")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--test_out", type=str, required=True, help="Path to test.out file")
    parser.add_argument("--class_names", type=str, required=True, help="Path to class_names.txt")
    parser.add_argument("--output_dir", type=str, default="visualize", help="Output directory for plots")
    parser.add_argument("--train_log", type=str, default=None, help="Optional: training log file for additional data")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("CNN RESULTS VISUALIZATION")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.results_csv)
    class_names = load_class_names(args.class_names)
    metrics, per_class_data = parse_test_output(args.test_out)
    
    print(f"✓ Loaded {len(df)} training epochs")
    print(f"✓ Loaded {len(class_names)} classes")
    print(f"✓ Loaded {len(per_class_data)} per-class metrics\n")
    
    # Generate plots
    print("Generating visualizations...\n")
    
    plot_training_loss(df, output_dir)
    plot_training_accuracy(df, output_dir)
    plot_combined_metrics(df, output_dir)
    
    if per_class_data:
        plot_per_class_f1(per_class_data, output_dir)
        plot_class_performance_comparison(per_class_data, output_dir)
        plot_top_bottom_classes(per_class_data, output_dir, n=10)
        plot_class_support_distribution(per_class_data, output_dir)
        plot_confusion_matrix_from_per_class(per_class_data, class_names, output_dir)
        
        # Class imbalance analysis
        plot_class_imbalance_analysis(per_class_data, output_dir)
        plot_performance_by_class_size(per_class_data, output_dir)
        plot_majority_minority_comparison(per_class_data, output_dir)
        plot_minority_class_focus(per_class_data, output_dir)
    else:
        print("⚠ Warning: No per-class data found, skipping per-class plots")
    
    plot_learning_rate_schedule(df, output_dir)
    plot_training_phases(df, output_dir)
    create_summary_report(df, metrics, per_class_data, output_dir)
    
    print("\n" + "="*80)
    print(f"All visualizations saved to: {output_dir.absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
