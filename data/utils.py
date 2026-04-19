from .dataset import SpeechCommandsDataset, Compose
from .transforms import MelSpectrogram, MFCC, Normalize, Flatten, FeatureConfig

def get_datasets(data_format: str, cfg: FeatureConfig | None = None, train_path="data/train", valid_path="data/valid") -> tuple[SpeechCommandsDataset, SpeechCommandsDataset]:
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

    Returns:
        - train_dataset: SpeechCommandsDataset for training
        - valid_dataset: SpeechCommandsDataset for validation
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

    return train_ds, valid_ds
    