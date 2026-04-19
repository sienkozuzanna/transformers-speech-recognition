import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    CNN baseline for audio classification.
    Input: [B, 1, n_mels/n_mfcc, T]  (mel or mfcc 2D feature map)
    Output: [B, num_classes]

    Simple CNN with 3 convolutional blocks.

    Architecture:
        Conv2D -> Batch Normalization -> ReLU -> MaxPool  (x3)
        AdaptiveAvgPool -> Flatten -> Dropout -> Linear
    """

    def __init__(self, num_classes: int = 12, in_channels: int = 1, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        channels1, channels2, channels3 = base_channels, base_channels * 2, base_channels * 4

        self.features = nn.Sequential(
            conv_block(in_channels, channels1), # [B, 32, H/2, W/2]
            conv_block(channels1, channels2), # [B, 64, H/4, W/4]
            conv_block(channels2, channels3),  # [B, 128, H/8, W/8]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # [B, 128, 1, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
