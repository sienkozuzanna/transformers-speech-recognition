import torch.nn as nn
import torch

class Transformer(nn.Module):
    """
    Baseline Transfoermer model for speech command recognition.
    Input: [B, 1, n_features, T]
    Works for MFCC [B, 1, 40, 101] and Mel [B, 1, 64, 101]
    """
    def __init__(self, n_features=40, n_timesteps=101, num_classes=12, d_model=128, nhead=4, num_layers=4, dropout=0.1, pooling='mean'):
        super().__init__()

        """
        Args:
            n_features: Number of input features (e.g., 40 for MFCC, 64 for Mel)
            n_timesteps: Number of time steps (e.g., 101 for 1-second audio with 10ms hop)
            num_classes: Number of output classes, in this case 12 (10 commands + unknown + silence)
            d_model: Dimension of the model (embedding size)
            nhead: Number of attention heads
            num_layers: Number of Transformer encoder layers
            dropout: Dropout rate
            pooling: Type of pooling to apply after the Transformer ('mean' or 'attention_pooling_linear', 'attention_pooling_sequential')
        """

        if pooling not in ['mean', 'attention_pooling_linear', 'attention_pooling_sequential']:
            raise ValueError("Pooling must be one of 'mean', 'attention_pooling_linear', 'attention_pooling_sequential'")
        self.pooling = pooling

        if self.pooling == 'attention_pooling_linear':
            self.attention_pooling = nn.Linear(d_model, 1) #learnable attention pooling weights
        elif self.pooling == 'attention_pooling_sequential':
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            )
        else:
            self.attention_pooling = None


        #input embedding - projection of input features to d_model dimension
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_timesteps, d_model)) #learnable positional encoding

        #single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, dim_feedforward= 4*d_model, dropout=dropout, batch_first=True)
        #stack of transformer encoder layers (num_layers)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), #normalization to prevent vanishing/exploding gradients
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1) # -> [B, n_features, T]
        x = x.permute(0, 2, 1) #-> [B, T, n_features], because of batch_first = True in transformer
        x = self.input_projection(x) # -> [B, T, d_model]
        x = x + self.positional_encoding[:, :x.size(1), :] #add positional encoding
        x = self.transformer(x) # -> [B, T, d_model]
        if self.pooling == 'mean':
            x = x.mean(dim = 1) #global average pooling over time dimension, to get one representation per sample -> [B, d_model]
        elif self.pooling in ['attention_pooling_linear', 'attention_pooling_sequential']:
            attn = self.attention_pooling(x) # [B, T, 1]
            attn = torch.softmax(attn, dim=1) # [B, T, 1]
            x = (x * attn).sum(dim=1) # idea: the embedding are weighted by the attention weights (more important features get higher weights) -> [B, d_model]
        else:
            raise ValueError("Invalid pooling type")
        x = self.classifier(x) # -> [B, num_classes]
        return x

