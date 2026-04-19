from .dataset import SpeechCommandsDataset, Compose
from .transforms import MelSpectrogram, MFCC, Normalize, Flatten, FeatureConfig
import torch
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset

def get_datasets(data_format: str, cfg: FeatureConfig | None = None, train_path="data/train", valid_path="data/valid", test_path="data/test") -> tuple[SpeechCommandsDataset, SpeechCommandsDataset]:
    """
    Returns train and valid datasets with the appropriate transforms based on the data_format.

    Args:
    - data_format: str, one of "raw", "mel", "mfcc", "mlr"
         data_format:
        - "raw"   -> waveform [1, 16000]
        - "mel"   -> MelSpectrogram + Normalize
        - "mfcc"  -> MFCC + Normalize
        - "mfcc_lr" -> MFCC + Flatten (logistic regression)
    - cfg: FeatureConfig, optional configuration for the transforms 
    - train_path: str, path to the training data directory
    - valid_path: str, path to the validation data directory
    - test_path: str, path to the test data directory

    Returns:
        - train_dataset: SpeechCommandsDataset for training
        - valid_dataset: SpeechCommandsDataset for validation
        - test_dataset: SpeechCommandsDataset for testing
    """

    if cfg is None:
        cfg = FeatureConfig()
   
    if data_format == "raw":
        transform = None

    elif data_format == "mel":
        transform = Compose([
            MelSpectrogram(cfg),
            Normalize()
        ])

    elif data_format == "mfcc":
        transform = Compose([
            MFCC(cfg),
            Normalize()
        ])

    elif data_format == "mfcc_lr":
        transform = Compose([
            MFCC(cfg),
            Flatten()
        ])

    else:
        raise ValueError(f"Unknown format: {data_format}")
    

    train_ds = SpeechCommandsDataset(train_path, cfg =cfg, transform=transform)
    valid_ds = SpeechCommandsDataset(valid_path, cfg =cfg, transform=transform)
    test_ds = SpeechCommandsDataset(test_path, cfg =cfg, transform=transform)

    return train_ds, valid_ds, test_ds


def precompute_features(dataset, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    for i, (x, y) in enumerate(tqdm(dataset)):
        torch.save((x, y), save_path / f'{i}.pt')
    print(f"Saved {len(dataset)} samples to {save_path}")

class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(Path(cache_dir).glob('*.pt'), key=lambda p: int(p.stem))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        return torch.load(self.files[i], weights_only=True)
    