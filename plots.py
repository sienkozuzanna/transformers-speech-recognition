import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data.dataset import CLASSES

PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#E84855',
    'accent':'#F5A623',
    'success':'#44BBA4',
    'bg': '#FFFFFF',
    'grid':'#E8E8E8',
}

plt.rcParams.update({
    'figure.facecolor': PALETTE['bg'],
    'axes.facecolor': PALETTE['bg'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color':  PALETTE['grid'],
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.frameon': False,
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
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes,
        vmin=0, vmax=vmax, linewidths=0.5, linecolor=PALETTE['grid'], ax=ax)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=16)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
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
        'Recall': [report[c]['recall'] for c in classes],
        'F1': [report[c]['f1-score'] for c in classes],
    }
    colors = [PALETTE['primary'], PALETTE['secondary'], PALETTE['success']]

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

def plot_training_curves(history, title='Training Curves'):
    """
    Plot loss and accuracy curves for a single training run.

    Args:
        history: dict with keys 'train_loss', 'valid_loss', 'train_acc', 'valid_acc' (lists of values per epoch)
        title: plot title
    """

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    ax1.plot(epochs, history['train_loss'], color=PALETTE['primary'],  linewidth=2, label='Train')
    ax1.plot(epochs, history['valid_loss'], color=PALETTE['secondary'], linewidth=2, label='Valid', linestyle='--')
    ax1.set_title('Loss');  ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], color=PALETTE['primary'],  linewidth=2, label='Train')
    ax2.plot(epochs, history['valid_acc'], color=PALETTE['secondary'], linewidth=2, label='Valid', linestyle='--')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_training_curves_multiseed(histories, title='Training Curves (mean +- std)'):
    """
    Plot mean +- std loss and accuracy across multiple seeds.

    Args:
        histories: list of history dicts (one per seed), each with keys 'train_loss', 'valid_loss', 'train_acc', 'valid_acc'
        title: plot title
    """

    keys = ['train_loss', 'valid_loss', 'train_acc', 'valid_acc']
    stacked = {k: np.array([h[k] for h in histories]) for k in keys}
    epochs = range(1, stacked['train_loss'].shape[1] + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    for ax, loss_key, acc_key, subtitle in [(ax1, 'train_loss', None, 'Loss'), (ax2, 'valid_loss', None, 'Loss')]:
        pass

    pairs = [(ax1, 'train_loss', 'valid_loss', 'Loss'), (ax2, 'train_acc',  'valid_acc',  'Accuracy'),]
    for ax, train_key, valid_key, ylabel in pairs:
        for key, color, label, ls in [(train_key, PALETTE['primary'],   'Train', '-'), (valid_key, PALETTE['secondary'], 'Valid', '--')]:
            mean = stacked[key].mean(axis=0)
            std = stacked[key].std(axis=0)
            ax.plot(epochs, mean, color=color, linewidth=2, label=label, linestyle=ls)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if ylabel == 'Accuracy':
            ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict, metric='macro_f1', title=None):
    """
    Bar chart comparing multiple models on a single metric.

    Args:
        results_dict: dict mapping model_name -> evaluate() output dict e.g. {'CNN': result_cnn, 'Transformer': result_tr, ...}
        metric: key from evaluate() output: 'acc', 'macro_f1', 'weighted_f1'
        title: plot title (auto-generated if None)
    """

    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    colors = sns.color_palette('deep', n_colors=len(models))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.6, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ylabel = {'acc': 'Accuracy', 'macro_f1': 'Macro F1', 'weighted_f1': 'Weighted F1'}.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.set_ylim(max(0, min(values) - 0.05), min(1.0, max(values) + 0.06))
    ax.set_title(title or f'Model Comparison – {ylabel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()