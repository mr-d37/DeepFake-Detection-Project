import torch
import torch.nn as nn
import torchvision.models as models


class DeepFakeDetector(nn.Module):
    """
    Deep Fake Detection model combining ResNet50 (pretrained) for feature extraction
    and LSTM for sequence classification.

    Architecture:
    - ResNet50 (pretrained on ImageNet): Extracts features from each frame
    - LSTM: Learns temporal patterns across frame sequences
    - Fully connected layers: Final classification
    """

    def __init__(self, num_classes=2, hidden_size=512, num_lstm_layers=2, dropout=0.5):
        """
        Initialize the DeepFakeDetector model.

        Args:
            num_classes: Number of output classes (2 for real/fake)
            hidden_size: Hidden state size for LSTM
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
        """
        super(DeepFakeDetector, self).__init__()

        resnet = models.resnext50_32x4d(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_dim = resnet.fc.in_features

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, num_frames, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size, num_frames, c, h, w = x.size()

        x = x.view(batch_size * num_frames, c, h, w)

        features = self.feature_extractor(x)

        features = features.view(batch_size, num_frames, -1)

        lstm_out, (h_n, c_n) = self.lstm(features)

        final_feature = lstm_out[:, -1, :]

        output = self.classifier(final_feature)

        return output


def create_model(num_classes=2, hidden_size=512, num_lstm_layers=2, dropout=0.5):
    """
    Factory function to create a DeepFakeDetector model.

    Args:
        num_classes: Number of output classes
        hidden_size: LSTM hidden state size
        num_lstm_layers: Number of LSTM layers
        dropout: Dropout probability

    Returns:
        DeepFakeDetector model instance
    """
    return DeepFakeDetector(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout
    )


if __name__ == "__main__":
    model = create_model()

    dummy_input = torch.randn(2, 40, 3, 224, 224)

    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")