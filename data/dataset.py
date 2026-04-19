import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from pathlib import Path
from typing import Optional, Callable

from .transforms import FeatureConfig

#label mapping
TARGET_WORDS = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}
CLASSES = sorted(TARGET_WORDS) + ['silence', 'unknown']   # 12 classes
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def map_label(word: str) -> str:
    if word == 'silence': return 'silence'
    if word in TARGET_WORDS: return word
    return 'unknown'

#Composes tranforms together
class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        steps = "\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {steps}\n])"


#main dataset class
class SpeechCommandsDataset(Dataset):
    """
    Loads .wav files and returns (x, target).
   
    x - result of transform(waveform) if transform is not None, otherwise raw waveform as [1, n_samples] tensor.
    target - int, class index in CLASSES (0-11)
    """

    def __init__(self, split_dir : str | Path, cfg: FeatureConfig = None, transform: Optional[Callable] = None,):
        self.split_dir = Path(split_dir)
        self.cfg = cfg or FeatureConfig()
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        for word_dir in sorted(self.split_dir.iterdir()):
            if not word_dir.is_dir():
                continue
            label = map_label(word_dir.name)
            target = CLASS_TO_IDX[label]
            for wav_path in sorted(word_dir.glob('*.wav')):
                self.samples.append((wav_path, target))

    def _load_waveform(self, path: Path) -> torch.Tensor:
        y, _ = librosa.load(str(path), sr=self.cfg.sample_rate, mono=True)
        y = librosa.util.fix_length(y, size=self.cfg.num_samples)
        return torch.from_numpy(y).unsqueeze(0).float() # [1, n_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        wav_path, target = self.samples[idx]
        x = self._load_waveform(wav_path)

        if self.transform is not None:
            x = self.transform(x)

        return x, target

    #sklearn helper
    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (X, y) as numpy arrays - ready for sklearn for logistic regression.
        Requires transform that returns 1D tensor
        """
        X_list, y_list = [], []
        for idx in range(len(self)):
            x, target = self[idx]
            if x.dim() != 1:
                raise ValueError(f"to_arrays() requires transform that returns 1D tensor, but got shape {x.shape}. Consider adding a Flatten() transform at the end of your transform pipeline.")
            X_list.append(x.numpy())
            y_list.append(target)

        return np.stack(X_list), np.array(y_list, dtype=np.int64)

    @staticmethod
    def label_name(idx: int) -> str:
        return CLASSES[idx]

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in CLASSES}
        for _, target in self.samples:
            counts[CLASSES[target]] += 1
        return counts
    

class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(Path(cache_dir).glob('*.pt'), key=lambda p: int(p.stem))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        return torch.load(self.files[i], weights_only=True)