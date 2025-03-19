import torch.nn as nn
import torchvision.models as models


class TSN(nn.Module):
    def __init__(self, num_classes=10, num_segments=3):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Remove last FC layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, segments, C, H, W = x.shape  # (batch, num_segments, 3, 224, 224)
        x = x.view(batch_size * segments, C, H, W)  # Merge batch and segment dimensions
        x = self.feature_extractor(x)  # Feature extraction
        x = x.view(batch_size, segments, -1).mean(dim=1)  # Temporal average pooling
        x = self.fc(x)  # Classification
        return x
