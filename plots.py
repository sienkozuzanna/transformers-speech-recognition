import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data.dataset import CLASSES

PALETTE = {
    'train':   '#2E86AB',
    'valid':   '#E84855',
    'accent':  '#F5A623',
    'success': '#44BBA4',
    'bg':      '#FFFFFF',
    'grid':    '#E0E0E0',
}

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['bg'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        PALETTE['grid'],
    'grid.linestyle':    '--',
    'grid.alpha':        0.7,
    'font.family':       'sans-serif',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'legend.frameon':    False,
})


def plot_confusion_matrix(cm, classes=CLASSES, normalize=True, title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap.

    Args:
        cm: confusion_matrix output from evaluate()
        classes: list of class names (default: CLASSES from dataset.py)
        normalize: if True, show percentages instead of counts
        title: plot title
    """
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt, vmax = '.2f', 1.0
    else:
        cm_plot = cm
        fmt, vmax = 'd', None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        vmin=0, vmax=vmax,
        linewidths=1, linecolor=PALETTE['grid'],
        ax=ax,
    )
    ax.set_title(title, fontsize=15, fontweight='bold', pad=16)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_per_class_metrics(labels, preds, classes=CLASSES, title='Per-class Metrics'):
    """
    Bar chart of per-class precision, recall, and F1.
 
    Args:
        labels: true labels (numpy array)
        preds: predicted labels (numpy array)
        classes: list of class names
        title: plot title
    """
    report = classification_report(labels, preds, target_names=classes, output_dict=True)
 
    metrics = {
        'Precision': [report[c]['precision'] for c in classes],
        'Recall':    [report[c]['recall']    for c in classes],
        'F1':        [report[c]['f1-score']  for c in classes],
    }
    colors = [PALETTE['train'], PALETTE['valid'], PALETTE['success']]
 
    x = np.arange(len(classes))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 5))
 
    for i, (name, vals) in enumerate(metrics.items()):
        ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.9)
 
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=40, ha='right')
    ax.set_ylabel('Score')
    ax.set_xlabel('Class')
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title='Training History'):
    """
    Plot loss and accuracy curves for a single training run.
    Best validation accuracy epoch is highlighted on the accuracy plot.

    Args:
        history: dict with keys 'train_loss', 'valid_loss', 'train_acc', 'valid_acc'
                 (lists of values per epoch)
        title: plot title
    """
    epochs = range(1, len(history['train_loss']) + 1)
    best_epoch = int(np.argmax(history['valid_acc'])) + 1
    best_acc = max(history['valid_acc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)

    ax1.plot(epochs, history['train_loss'], color=PALETTE['train'],  linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['valid_loss'], color=PALETTE['valid'],  linewidth=2, linestyle='--', label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, history['valid_acc'], color=PALETTE['valid'], linewidth=2, label='Validation Accuracy')
    ax2.axvline(best_epoch, color='gray', linestyle=':', linewidth=1.5)
    ax2.scatter([best_epoch], [best_acc], color=PALETTE['valid'], zorder=5)
    ax2.annotate(f'best: {best_acc:.3f}',
                 xy=(best_epoch, best_acc),
                 xytext=(8, -15), textcoords='offset points',
                 fontsize=9, color='gray')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_training_curves_multiseed(histories, title='Training History (mean ± std)'):
    """
    Plot mean ± std loss and accuracy across multiple seeds.
    Best mean validation accuracy epoch is highlighted on the accuracy plot.

    Args:
        histories: list of history dicts (one per seed), each with keys
                   'train_loss', 'valid_loss', 'train_acc', 'valid_acc'
        title: plot title
    """
    keys = ['train_loss', 'valid_loss', 'train_acc', 'valid_acc']
    stacked = {k: np.array([h[k] for h in histories]) for k in keys}
    epochs = range(1, stacked['train_loss'].shape[1] + 1)

    mean_valid_acc = stacked['valid_acc'].mean(axis=0)
    best_epoch = int(np.argmax(mean_valid_acc)) + 1
    best_acc = mean_valid_acc[best_epoch - 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)

    for key, color, label, ls in [
        ('train_loss', PALETTE['train'], 'Training Loss',   '-'),
        ('valid_loss', PALETTE['valid'], 'Validation Loss', '--'),
    ]:
        mean = stacked[key].mean(axis=0)
        std = stacked[key].std(axis=0)
        ax1.plot(epochs, mean, color=color, linewidth=2, label=label, linestyle=ls)
        ax1.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    mean = stacked['valid_acc'].mean(axis=0)
    std = stacked['valid_acc'].std(axis=0)

    ax2.plot(epochs, mean, color=PALETTE['valid'], linewidth=2, label='Validation Accuracy')
    ax2.fill_between(epochs, mean - std, mean + std, color=PALETTE['valid'], alpha=0.15)
    ax2.axvline(best_epoch, color='gray', linestyle=':', linewidth=1.5)
    ax2.scatter([best_epoch], [best_acc], color=PALETTE['valid'], zorder=5)
    ax2.annotate(f'best: {best_acc:.3f}',
                 xy=(best_epoch, best_acc),
                 xytext=(8, -15), textcoords='offset points',
                 fontsize=9, color='gray')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict, metric='macro_f1', title=None):
    """
    Bar chart comparing multiple models on a single metric.

    Args:
        results_dict: dict mapping model_name -> evaluate() output dict
                      e.g. {'CNN': result_cnn, 'Transformer': result_tr, ...}
        metric: key from evaluate() output: 'acc', 'macro_f1', 'weighted_f1'
        title: plot title (auto-generated if None)
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    colors = sns.color_palette('deep', n_colors=len(models))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.6, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10,
        )

    ylabel = {'acc': 'Accuracy', 'macro_f1': 'Macro F1', 'weighted_f1': 'Weighted F1'}.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.set_ylim(max(0, min(values) - 0.05), min(1.0, max(values) + 0.06))
    ax.set_title(title or f'Model Comparison – {ylabel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()