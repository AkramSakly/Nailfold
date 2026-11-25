"""
model.py
--------
CE-NFCNet model, following the paper:
  Ebadi Jalal et al. (2025), "Abnormality detection in nailfold capillary
  images using deep learning with EfficientNet and cascade transfer learning",
  Scientific Reports 15:2068.

Base network: EfficientNet-B0, pre-trained on ImageNet.

Classifier head (paper, Fig. 3 / Stage 8):
    Dropout(0.3) -> GlobalAveragePooling2D -> Dropout(0.3)
    -> BatchNormalization -> Dense(num_classes)

(torchvision's EfficientNet already applies global average pooling internally
in `avgpool`, so we replace the default classifier with the regularization
head above, operating on the 1280-d pooled feature vector.)

Fine-tuning protocol:
  The paper freezes the backbone and fine-tunes only the last stages/blocks.
  In torchvision's EfficientNet-B0 the feature extractor is `features`, a
  Sequential of 9 blocks (indices 0..8). The paper fine-tunes the deeper
  stages (6, 7, 8). We expose `set_trainable_stages()` to freeze everything
  except the chosen stage indices plus the classifier head, matching the
  cascade idea (Step A: stages 6,7,8 ; Step B: stages 7,8).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class CENFCNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Feature extractor: conv stem + MBConv stages (indices 0..8)
        self.features = backbone.features      # Sequential of 9 blocks
        self.avgpool = backbone.avgpool        # global average pooling -> 1280

        in_feats = backbone.classifier[1].in_features  # 1280 for B0

        # Paper's regularization head (Stage 8).
        # Dropout before pooling is applied to the feature map; pooling is done
        # by self.avgpool; then Dropout -> BatchNorm -> Dense.
        self.pre_pool_dropout = nn.Dropout2d(p=dropout)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(in_feats),
            nn.Linear(in_feats, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pre_pool_dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    # ------------------------------------------------------------------ #
    #  Fine-tuning control                                               #
    # ------------------------------------------------------------------ #
    def freeze_all_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def set_trainable_stages(self, stage_indices):
        """
        Freeze the whole backbone, then unfreeze only the given stage indices.
        The classifier head and the pre-pool dropout's following layers are
        always trainable. `stage_indices` refers to indices into
        `self.features` (0..8). Example: [6, 7, 8] for Step A, [7, 8] for Step B.
        """
        self.freeze_all_backbone()
        for idx in stage_indices:
            for p in self.features[idx].parameters():
                p.requires_grad = True
        # Head is always trainable
        for p in self.head.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick sanity check
    m = CENFCNet(num_classes=2, pretrained=False)
    m.set_trainable_stages([7, 8])
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("output shape:", y.shape)
    print("trainable params:", m.num_trainable())