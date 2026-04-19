import torch.nn as nn
import torch

class CNNTransformer(nn.Module):
    """
    CNN as feature extractor, Transformer as sequence model.
    Input: [B, 1, n_mfcc, T]
    Output: [B, num_classes]
    """

    def __init__(self, n_features=40, n_timesteps=101, num_classes=12, base_channels = 32, d_model=128, nhead=4, num_layers=4, dropout=0.1, pooling="mean"):
        """
        Args:
            n_features: Number of input features (e.g., 40 for MFCC, 64 for Mel)
            n_timesteps: Number of time steps (e.g., 101 for 1-second audio with 10ms hop)
            num_classes: Number of output classes, in this case 12 (10 commands + unknown + silence)
            base_channels: Number of channels in the first CNN layer, will be doubled in each subsequent layer
            d_model: Dimension of the model (embedding size)
            nhead: Number of attention heads
            num_layers: Number of Transformer encoder layers
            dropout: Dropout rate
            pooling: Type of pooling to apply after the Transformer ('mean' or 'attention_pooling_linear', 'attention_pooling_sequential')
        """

        super().__init__()
        if pooling not in ['mean', 'attention_pooling_linear', 'attention_pooling_sequential']:
            raise ValueError("Pooling must be one of 'mean', 'attention_pooling_linear', 'attention_pooling_sequential'")

        self.pooling = pooling

        #CNN feature extractor 
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
            )
        
        channels1, channels2, channels3 = base_channels, base_channels * 2, base_channels * 4

        self.cnn = nn.Sequential(
            conv_block(1, channels1), # [B, 32, H/2, W/2]
            conv_block(channels1, channels2), # [B, 64, H/4, W/4]
            conv_block(channels2, channels3),  # [B, 128, H/8, W/8]
        )

        cnn_out_freq = n_features // 8
        cnn_out_dim  = channels3 * cnn_out_freq

        #transformer input projection - to project the CNN output features to d_model dimension
        self.input_projection  = nn.Linear(cnn_out_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_timesteps, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        if pooling == 'attention_pooling_linear':
            self.attention_pooling = nn.Linear(d_model, 1)
        elif pooling == 'attention_pooling_sequential':
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            )
        else:
            self.attention_pooling = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, n_features, T]
        x = self.cnn(x) # [B, channels3, n_features/8, T]

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2) # [B, T, C, F]
        x = x.reshape(B, T, C * F)  # [B, T, C*F]

        x = self.input_projection(x) # [B, T, d_model]
        x = x + self.positional_encoding[:, :x.size(1), :] # add positional encoding
        x = self.transformer(x) # [B, T, d_model]

        if self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling in ['attention_pooling_linear', 'attention_pooling_sequential']:
            attn = self.attention_pooling(x) # [B, T, 1]
            attn = torch.softmax(attn, dim=1) # [B, T, 1]
            x = (x * attn).sum(dim=1) # [B, d_model]
        else:
            raise ValueError("Invalid pooling type")

        x = self.classifier(x) # [B, num_classes]
        return x