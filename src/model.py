import torch
import torch.nn as nn
from torchvision import models

class MultiTaskResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet18 without pre-trained weights (as per notebook, though transfer learning was mentioned in prompt, 
        # the notebook code had weights=None. The prompt said "train a pretrained ResNet-18", so I should probably change this to use weights.
        # However, to strictly follow the "refactor" instruction first, I will stick to the notebook logic but maybe add a comment or option.
        # Wait, the user prompt said "train a pretrained ResNet-18 adapted to 5-channel input". 
        # The notebook had `weights=None`. I will switch to `weights='DEFAULT'` or `weights=ResNet18_Weights.DEFAULT` to follow the USER REQUEST 
        # better than the notebook code, as the user explicitly asked for "pretrained".
        
        # Actually, let's stick to the notebook for now to ensure reproducibility of what they had, 
        # BUT the user request explicitly said "train a pretrained ResNet-18". 
        # I will use `weights=models.ResNet18_Weights.DEFAULT` to satisfy the user request.
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify input layer to accept 5 channels
        # The original first layer is: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # We need 5 input channels. We can average the weights of the first 3 channels to initialize the new 5 channels 
        # or just reset them. Resetting is safer/easier if we don't want to mess with weight copying logic right now.
        # But for "transfer learning", keeping weights is good. 
        # A common trick is to copy the weights.
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new conv1 weights
        with torch.no_grad():
            # Copy weights from the first 3 channels to the first 3 of the new conv
            self.backbone.conv1.weight[:, :3] = original_conv1.weight
            # For the remaining 2, we can reuse the first 2 or average. Let's just reuse the first 2 for simplicity/initialization.
            self.backbone.conv1.weight[:, 3:] = original_conv1.weight[:, :2]

        nfeat = self.backbone.fc.in_features

        # Remove final FC
        self.backbone.fc = nn.Identity()

        # 3 prediction heads:
        self.fc_shot  = nn.Linear(nfeat, 2)  # good/bad
        self.fc_stress = nn.Linear(nfeat, 1) # regression
        self.fc_task  = nn.Linear(nfeat, 3)  # 3-class state

    def forward(self, x):
        f = self.backbone(x)
        return (
            self.fc_shot(f),
            self.fc_stress(f),
            self.fc_task(f)
        )
