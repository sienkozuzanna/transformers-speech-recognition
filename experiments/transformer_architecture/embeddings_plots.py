from torch.utils.data import DataLoader
import torch
import numpy as np
from models.transformer import Transformer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from plots import PALETTE

def extract_embeddings(model, dataset, device, batch_size=256):
    """
    Extracts embeddings from the model's transformer layers for the given dataset.

    Args:
        - model: the trained Transformer model
        - dataset: the dataset to extract embeddings from
        - device: the device to run the model on
        - batch_size: the batch size for processing the dataset

    Returns:
        - all_embs: numpy array of shape [N, d_model] containing the extracted embeddings
        - all_labels: numpy array of shape [N] containing the corresponding labels
    """

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embs, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # forward step-by-step without final classification head to get embeddings
            h = x.squeeze(1).permute(0, 2, 1) # [B, T, F]
            h = model.input_projection(h)# [B, T, d_model]
            h = h + model.positional_encoding[:, :h.size(1), :]
            h = model.transformer(h) # [B, T, d_model]

            if model.pooling == 'mean':
                emb = h.mean(dim=1)
            else:
                attn = model.attention_pooling(h)
                attn = torch.softmax(attn, dim=1)
                emb  = (h * attn).sum(dim=1)

            all_embs.append(emb.cpu().numpy())
            all_labels.append(y.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)

def get_best_model(results, d_model):
    """Returns the best model for a given d_model based on test_macro_f1."""
    best_seed = max(results[d_model], key=lambda s: results[d_model][s]['test_macro_f1'])
    return results[d_model][best_seed]['model']

POOLING_SHORT = {
    'mean': 'mean',
    'attention_pooling_linear': 'attn_linear',
    'attention_pooling_sequential': 'attn_sequential',
}

def plot_tsne_comparison(embeddings_per_key, class_names, key_label='d_model',
                         title='t-SNE of embeddings', perplexity=40, random_state=42, sample_size=2000):
    """
    Plots t-SNE visualizations of embeddings side by side.
    Works for any experiment — d_model, pooling experiment, etc.

    Args:
        - embeddings_per_key: dict mapping any key to {'embs': [N, d_model], 'labels': [N]}
        - class_names: list of class names corresponding to label indices
        - key_label: label for the experiment dimension (e.g. 'd_model' or 'pooling')
        - title: suptitle of the figure
        - perplexity: t-SNE perplexity parameter
        - random_state: random seed for reproducibility
        - sample_size: number of samples for silhouette score computation
    """

    n = len(embeddings_per_key)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    colors = cm.tab20(np.linspace(0, 1, len(class_names)))

    for ax, (key, data) in zip(axes, embeddings_per_key.items()):
        embs = data['embs']
        labels = data['labels']

        n_pca = min(50, embs.shape[1])
        reduced_pca = PCA(n_components=n_pca, random_state=random_state).fit_transform(embs)

        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(embs), size=min(sample_size, len(embs)), replace=False)
        sil = silhouette_score(reduced_pca[idx], labels[idx])

        reduced_2d = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_iter=1000).fit_transform(reduced_pca)

        for cls_id, (cls_name, color) in enumerate(zip(class_names, colors)):
            mask = labels == cls_id
            ax.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                       color=color, alpha=0.55, s=12, linewidths=0)

        short_key = POOLING_SHORT.get(str(key), str(key))
        ax.set_title(f'{key_label} = {short_key}\nSilhouette = {sil:.3f}',
                     fontsize=13, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=colors[i],
               markersize=8, label=name)
        for i, name in enumerate(class_names)
    ]
    fig.legend(handles=handles, title='Klasa', loc='center right',
               bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()