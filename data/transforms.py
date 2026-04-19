import numpy as np
import torch
import librosa
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    duration: float = 1.0

    #mel=spectogram
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    fmin: int = 20
    fmax: int = 8000

    #MFCC
    n_mfcc: int = 40

    @property
    def num_samples(self):
        return int(self.sample_rate * self.duration)
    

class Waveform:
    '''
    A simple transform that returns the raw waveform as a tensor.
    This can be used as a placeholder when no feature extraction is needed, or when the model is designed to work directly with raw audio.
    '''

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
 
    def __repr__(self) -> str:
        return "Waveform()"

class MelSpectrogram:
    '''
    A transform that computes the Mel Spectrogram from the raw audio waveform.
    '''

    def __init__(self, cfg: FeatureConfig=None):
        self.cfg = cfg or FeatureConfig()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.squeeze(0).numpy()
        cfg = self.cfg
        mel = librosa.feature.melspectrogram(
            y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
        )
        mel_dbc = librosa.power_to_db(mel, ref = np.max).astype(np.float32)
        return torch.from_numpy(mel_dbc).unsqueeze(0)
    
    def __repr__(self) -> str:
        cfg = self.cfg
        return f"MelSpectrogram(n_mels={cfg.n_mels}, hop={cfg.hop_length})"
    

class MFCC:
    '''
    A transform that computes the Mel-Frequency Cepstral Coefficients (MFCCs) from the raw audio waveform.
    '''
    
    def __init__(self, cfg: FeatureConfig=None):
        self.cfg = cfg or FeatureConfig()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.squeeze(0).numpy()
        cfg = self.cfg
        mfcc = librosa.feature.mfcc(
            y=y, sr=cfg.sample_rate, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length, fmin=cfg.fmin, fmax=cfg.fmax
        ).astype(np.float32)
        return torch.from_numpy(mfcc).unsqueeze(0)
    
    def __repr__(self) -> str:
        cfg = self.cfg
        return f"MFCC(n_mfcc={cfg.n_mfcc}, hop={cfg.hop_length})"
    

class Normalize:
    '''
    A simple transform that normalizes the input tensor to have zero mean and unit variance.
    '''

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-6)
    
    def __repr__(self) -> str:
        return "Normalize()"
    
class Flatten:
    '''
    A simple transform that flattens the input tensor to a 1D vector.
    This can be useful when the model expects a flat input, or when we want to reduce the dimensionality of the features before feeding them into a classifier.
    '''

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten() 
    
    def __repr__(self) -> str:
        return "Flatten()"

