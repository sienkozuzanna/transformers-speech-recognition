# Speech Command Recognition with CNNs and Transformers

A deep learning project exploring spoken command classification using different neural network architectures and audio representations. The project investigates how model architecture, feature extraction, and training configurations affect classification performance on short audio recordings.

The experiments focus on comparing **MFCC and log-Mel spectrogram representations**, evaluating CNN-based and Transformer-based architectures, and analyzing model performance across multiple random seeds.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Audio Representations](#audio-representations)
- [Model Architectures](#model-architectures)
- [Experimental Setup](#experimental-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Visualization](#results-and-visualization)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
- [Technologies](#technologies)

## Overview

The objective of this project is to investigate deep learning approaches for recognizing short spoken commands from audio signals.

The experimental pipeline covers:

- Audio feature extraction and preprocessing.
- Comparison of MFCC and log-Mel spectrogram inputs.
- Implementation and evaluation of CNN, Transformer, and hybrid CNN–Transformer architectures.
- Hyperparameter experiments and comparisons across random seeds.
- Evaluation using accuracy and F1-score metrics.
- Visualization of training dynamics, model performance, and class-level results.

The project emphasizes systematic experimentation and reproducible comparisons rather than relying on a single model configuration.

## Dataset

The experiments use the **Google Speech Commands dataset**, designed for keyword spotting and spoken command classification.

The classification task includes 12 categories:

| Category | Labels |
|---|---|
| Spoken commands | `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go` |
| Additional categories | `unknown`, `silence` |

Each audio recording represents approximately one second of audio sampled at 16 kHz.

The prepared feature datasets contain the following numbers of examples:

| Split | Number of examples |
|---|---:|
| Training | 51,486 |
| Validation | 6,828 |
| Test | 6,865 |

The dataset is processed into numerical audio representations suitable for neural network training.

## Audio Representations

Two audio representations are investigated to compare the effects of different feature extraction approaches.

### MFCC — Mel-Frequency Cepstral Coefficients

MFCC features represent the short-term spectral envelope of an audio signal using a perceptually motivated frequency scale.

The project uses 40 MFCC coefficients over 101 time frames, producing a feature tensor of shape:

`(1, 40, 101)`

A flattened MFCC representation is also used, with 4,040 features per audio example.

### Log-Mel Spectrograms

Mel spectrograms represent the distribution of audio energy across Mel-scaled frequency bands over time. They provide a time-frequency representation that can be processed by convolutional and attention-based architectures.

The prepared Mel representation has the shape:

`(1, 64, 101)`

### Representation Comparison

Comparing MFCC and log-Mel inputs helps investigate how the information preserved by each representation affects classification quality across different neural architectures.

## Model Architectures

The project investigates three architectural approaches.

### Convolutional Neural Network (CNN)

The CNN serves as a convolution-based approach to audio classification. Convolutional layers learn local patterns in the time-frequency representation, while the classification head predicts the spoken command category.

### Transformer Encoder

The Transformer architecture uses self-attention to model relationships between positions in the audio feature sequence.

The investigated configuration includes:

- Embedding dimension (`d_model`): 128
- Number of attention heads (`nhead`): 4
- Number of Transformer encoder layers: 4

This architecture provides a comparison against convolutional models in terms of classification performance and training behavior.

### Hybrid CNN–Transformer

The hybrid architecture combines convolutional feature extraction with Transformer-based sequence modeling. The CNN component captures local time-frequency patterns, while self-attention models relationships across the resulting representation.

The hybrid approach allows the experiments to investigate whether combining local feature extraction and global sequence modeling improves classification performance.

## Experimental Setup

The experiments compare architectures, audio representations, and training configurations under a consistent evaluation workflow.

The main experimental dimensions include:

- **Architecture:** CNN, Transformer, and CNN–Transformer.
- **Input representation:** MFCC and log-Mel spectrograms.
- **Random seeds:** 42, 123, and 456 for experiments using multiple seeds.
- **Hyperparameters:** model-specific and training-related configuration choices.

Multiple random seeds allow performance variability to be assessed rather than reporting only a single run.

Validation data is used to monitor training performance and compare configurations. Test metrics are used to assess generalization to held-out examples.

## Evaluation Metrics

The evaluation and visualization utilities support the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Fraction of correctly classified examples |
| Macro F1-score | Unweighted average of F1-scores across classes |
| Weighted F1-score | Average F1-score weighted by class support |
| Training loss | Tracks optimization on the training set |
| Validation loss | Tracks performance on the validation set |

Macro F1 is particularly useful when evaluating command classes individually, as it gives each class equal weight regardless of its frequency.

For experiments involving multiple seeds, the analysis includes means and standard deviations. Coefficients of variation are also calculated where applicable.

## Results and Visualization

The repository includes `plots.py`, a collection of plotting and analysis utilities for comparing experimental results.

The available utilities cover:

- Training and validation loss curves.
- Validation F1-score curves across epochs.
- Model comparison plots.
- Accuracy, macro F1, and weighted F1 comparisons.
- Summary tables reporting mean and standard deviation across seeds.
- Heatmaps comparing models and audio representations.
- Per-class F1-score comparisons.
- Hyperparameter comparisons.
- Confusion-matrix-based class-level analysis.

The visualizations support analysis of overall classification quality, differences between audio representations, convergence behavior, and variability across random seeds.

The plotting functions are designed to work with structured experiment results, allowing results from different models and configurations to be consolidated into comparative summaries.

## Repository Structure

The repository is organized into data, experiments, model implementations, notebooks, results, and visualization utilities.

```text
transformers-speech-recognition/
├── data/           # Dataset and prepared audio features
├── experiments/    # Experimental configurations and runs
├── models/         # Neural network model implementations
├── notebooks/      # Exploratory analysis and experiments
├── results/        # Experimental outputs and evaluation results
├── plots.py        # Plotting and results analysis utilities
├── requirements.txt
└── README.md
```

The exact contents of individual directories may vary between experiments.

## Installation

The project uses Python and PyTorch for deep learning, together with libraries for audio processing, data analysis, and visualization.

Clone the repository:

```bash
git clone https://github.com/sienkozuzanna/transformers-speech-recognition.git
cd transformers-speech-recognition
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository's `requirements.txt` specifies the versions of its principal dependencies, including PyTorch, NumPy, pandas, scikit-learn, librosa, Matplotlib, and Seaborn.

## Running the Experiments

The repository contains Jupyter notebooks and experiment-related files. The appropriate notebook or experiment entry point should be selected based on the model and configuration being investigated.

To start Jupyter:

```bash
jupyter notebook
```

Open the relevant notebook from the `notebooks/` directory and execute its cells in order.

The plotting utilities can be imported into a notebook for result analysis:

```python
from plots import (
    plot_metrics_comparison,
    plot_summary_table,
    plot_f1_training_curves,
    plot_repr_model_heatmap,
)
```

These functions expect structured result dictionaries or summary DataFrames, depending on the selected visualization. Refer to their definitions in `plots.py` for the required input formats.

## Technologies

- **Python** — experiment implementation and analysis.
- **PyTorch** — neural network training.
- **librosa** — audio feature extraction.
- **NumPy and pandas** — numerical processing and result aggregation.
- **scikit-learn** — machine learning evaluation utilities.
- **Matplotlib and Seaborn** — experiment visualization.
- **Jupyter Notebook** — exploratory analysis and experimental workflows.
