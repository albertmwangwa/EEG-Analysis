import torch
import torch.nn as nn
from torchvision import models


class MultiTaskResNet(nn.Module):
    """ResNet-18 adapted to 5-channel EEG scalp images with three task heads.

    Input:  (N, 5, 64, 64) tensor, one channel per frequency band
            (Delta, Theta, Alpha, Beta, Gamma).
    Output: tuple of (shot logits [N, 2], stress [N, 1], task logits [N, 3]).
    """

    def __init__(self):
        super().__init__()
        # Start from ImageNet weights for transfer learning.
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Widen the first convolution from 3 to 5 input channels, reusing the
        # pretrained filters so early features are not learned from scratch.
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = original_conv1.weight
            self.backbone.conv1.weight[:, 3:] = original_conv1.weight[:, :2]

        nfeat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Task heads on the shared 512-d feature vector.
        self.fc_shot = nn.Linear(nfeat, 2)    # shot quality: good / bad
        self.fc_stress = nn.Linear(nfeat, 1)  # stress level (regression)
        self.fc_task = nn.Linear(nfeat, 3)    # task / cognitive state (3 classes)

    def forward(self, x):
        f = self.backbone(x)
        return (
            self.fc_shot(f),
            self.fc_stress(f),
            self.fc_task(f),
        )
