# layers.py

import torch.nn as nn

class ModeloCNN(nn.Module):
    def __init__(self, num_classes: int = 38):
        super().__init__()
        
        # === feature extractor ===
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 256→128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 128→64
        )
        
        # === classification head ===
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 64×64×64 → 64×1×1
            nn.Flatten(),                  # → [batch, 64]
            nn.Linear(64, num_classes)     # → [batch, num_classes]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
