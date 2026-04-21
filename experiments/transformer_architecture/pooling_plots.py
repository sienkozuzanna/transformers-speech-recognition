import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from plots import PALETTE

def extract_attention_weights(model, dataset, device, batch_size=256):
    """
    Returns the attention weights from the model's attention pooling layer for the given dataset.
    """
    assert model.pooling in ['attention_pooling_linear', 'attention_pooling_sequential'], "Model must have attention pooling!"

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_weights, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = x.squeeze(1).permute(0, 2, 1)
            h = model.input_projection(h)
            h = h + model.positional_encoding[:, :h.size(1), :]
            h = model.transformer(h) # [B, T, d_model]

            attn = model.attention_pooling(h) # [B, T, 1]
            attn = torch.softmax(attn, dim=1) # [B, T, 1]
            attn = attn.squeeze(-1) # [B, T]

            all_weights.append(attn.cpu().numpy())
            all_labels.append(y.numpy())

    return np.concatenate(all_weights), np.concatenate(all_labels)


def plot_attention_heatmap_comparison(attn_weights_per_pooling, class_names):
    """
    Plots two heatmaps side by side comparing the mean attention weights across timesteps for each class, for different pooling methods.

    Args:
        attn_weights_per_pooling: dict with keys being pooling method names and values being dicts with 'weights' and 'labels'
        class_names: list of class names corresponding to label indices
    """
    poolings = list(attn_weights_per_pooling.keys())

    #claculating mean weights for each pooling option
    mean_weights_all = {}
    for pooling, data in attn_weights_per_pooling.items():
        weights = data['weights']
        labels = data['labels']
        mean_weights_all[pooling] = np.array([weights[labels == cls_id].mean(axis=0) for cls_id in range(len(class_names))])

    vmin = min(mw.min() for mw in mean_weights_all.values())
    vmax = max(mw.max() for mw in mean_weights_all.values())

    fig, axes = plt.subplots(1, len(poolings), figsize=(14 * len(poolings) / 2 + 1, 6), sharey=True)
    if len(poolings) == 1:
        axes = [axes]

    for ax, pooling in zip(axes, poolings):
        mean_w = mean_weights_all[pooling]
        im = ax.imshow(mean_w, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=vmin, vmax=vmax)
        short = pooling.replace('attention_pooling_', 'attn_')
        ax.set_title(f'pooling = {short}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep (MFCC frames)', fontsize=11)

    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_yticklabels(class_names, fontsize=11)

    fig.colorbar(im, ax=axes[-1], label='Mean attention weight', fraction=0.02, pad=0.04)
    fig.suptitle('Attention pooling weights per class\n(lighter = higher weights = more important timestemps)', fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_attention_profiles(attn_weights_per_pooling, class_names, classes_to_show=None):
    """
    Line plot of attention profiles for selected classes, one line - one pooling option
    """
    if classes_to_show is None:
        classes_to_show = class_names

    cls_indices  = [class_names.index(c) for c in classes_to_show]
    poolings = list(attn_weights_per_pooling.keys())
    colors_pool = ['#4C72B0', '#DD8452']
    T = list(attn_weights_per_pooling.values())[0]['weights'].shape[1]
    timesteps = np.arange(T)

    n_cls = len(classes_to_show)
    n_cols = 2
    n_rows = (n_cls + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.5 * n_rows), sharex=True, sharey=False)
    axes_flat = axes.flatten() if n_cls > 1 else [axes]

    for ax, cls_name, cls_id in zip(axes_flat, classes_to_show, cls_indices):
        for color, pooling in zip(colors_pool, poolings):
            data = attn_weights_per_pooling[pooling]
            weights = data['weights'][data['labels'] == cls_id] # [N_cls, T]
            mean_w = weights.mean(axis=0)
            std_w = weights.std(axis=0)
            short = pooling.replace('attention_pooling_', 'attn_')

            ax.plot(timesteps, mean_w, color=color, lw=1.8, label=short)
            ax.fill_between(timesteps, mean_w - std_w, mean_w + std_w, color=color, alpha=0.12)

        ax.set_title(f'"{cls_name}"', fontsize=11, fontweight='bold')
        ax.set_ylabel('Weight', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes_flat[n_cls:]:
        ax.set_visible(False)

    for ax in axes_flat[-n_cols:]:
        ax.set_xlabel('Timestep (MFCC frame)', fontsize=10)

    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='upper right', bbox_to_anchor=(1.08, 0.98), frameon=False, fontsize=10, title='pooling')
    fig.suptitle('Attention profile per class — linear vs sequential', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()