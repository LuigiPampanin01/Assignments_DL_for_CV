import torch
import torch.nn as nn
import torchvision.models as models


class ResNetPatchClassifier(nn.Module):
    def __init__(
        self,
        n_object_classes: int = 2,          # pothole vs background
        backbone: str = "resnet18",         # "resnet18", "resnet34", ...
        pretrained: bool = True,
        binary_mode: str = "sofmax"         # "logits" or "softmax"
    ):
        """
        n_object_classes:
            For your case, 2 (pothole, background).
        backbone:
            Smaller (resnet18) = faster, good for 60k patches.
        pretrained:
            If True, load ImageNet weights and fine-tune.
        binary_mode:
            - "logits": output shape [B, 1], use BCEWithLogitsLoss.
            - "softmax": output shape [B, 2], use CrossEntropyLoss.
        """
        super().__init__()
        assert binary_mode in ["logits", "softmax"]

        self.binary_mode = binary_mode

        # ---- 1. Load backbone ----
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ---- 2. Replace final FC layer ----
        in_features = self.backbone.fc.in_features
        self.dropout = nn.Dropout2d(p=0.6)

        if self.binary_mode == "logits":
            # Single logit → use BCEWithLogitsLoss
            self.backbone.fc = nn.Linear(in_features, 1)
        else:
            # Two logits → use CrossEntropyLoss
            self.backbone.fc = nn.Linear(in_features, n_object_classes)

    def forward(self, x):
        """
        x: [B, 3, H, W] patches from your PotholeDataset
        """
        
        # 1. Forward through the main ResNet layers (up to Layer 4)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # 2. Global Average Pooling (ResNet's standard next step)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # --- NEW: Apply Dropout ---
        x = self.dropout(x) # <--- ADD THIS LINE

        # 3. Final Classification Layer
        out = self.backbone.fc(x)

        return out


