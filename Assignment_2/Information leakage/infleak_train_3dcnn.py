# train_c3d_simple.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import FrameVideoDataset          # <- uses videos: [C, T, H, W]
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json

writer = SummaryWriter(log_dir="runs/c3d_simple")

import numpy as np  # NEW

# --- temporal sampling config ---
T_TARGET = 16  # the temporal length your 3D model will always see (use 10 if OOM)

def temporal_jitter(x, t_target=16, max_stride=2, reverse_p=0.1):
    """
    x: [B, C, T, H, W]  -> randomly sample a contiguous clip of length t_target.
    We also sometimes reverse time for augmentation.
    """
    B, C, T, H, W = x.shape
    # occasional reverse of the whole sequence
    if reverse_p > 0:
        mask = (torch.rand(B, device=x.device) < reverse_p)
        x[mask] = x[mask].flip(dims=[2])

    out = []
    for b in range(B):
        stride = np.random.randint(1, max_stride+1) if max_stride >= 1 else 1
        need = t_target
        max_start = max(0, T - stride*(need-1) - 1)
        start = np.random.randint(0, max_start+1) if max_start > 0 else 0
        idxs = [min(start + i*stride, T-1) for i in range(need)]
        out.append(x[b:b+1, :, idxs, :, :])  # [1,C,t_target,H,W]
    return torch.cat(out, dim=0)

def temporal_center_crop(x, t_target=16):
    """Deterministic center crop in time for val/test."""
    B, C, T, H, W = x.shape
    if T >= t_target:
        s = (T - t_target)//2
        return x[:, :, s:s+t_target]
    # pad last frame if too short
    pad = t_target - T
    last = x[:, :, -1:].repeat(1,1,pad,1,1)
    return torch.cat([x, last], dim=2)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ The code will run on NVIDIA GPU (CUDA).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ The code will run on Apple Silicon GPU (MPS).")
else:
    device = torch.device("cpu")
    print("⚠️ The code will run on CPU.")

# ---------------- DATASETS ----------------
root_dir = '/dtu/datasets1/02516/ucf101_noleakage'
img_size = 64
batch_size = 16         # 3D needs a bit more memory; drop to 8 if OOM
num_epochs = 20
num_classes = 10

train_tf = T.Compose([
    T.Resize((img_size, img_size)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.ToTensor(),
    T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])

eval_tf = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])

# IMPORTANT: use the video dataset (returns [C, T, H, W])
trainset = FrameVideoDataset(root_dir=root_dir, split='train', transform=train_tf,   stack_frames=True)
valset   = FrameVideoDataset(root_dir=root_dir, split='val',   transform=eval_tf,    stack_frames=True)
testset  = FrameVideoDataset(root_dir=root_dir, split='test',  transform=eval_tf,    stack_frames=True)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ---------------- SIMPLE 3D CNN (C3D-ish, VGG-style) ----------------
class SimpleC3D(nn.Module):
    """
    Minimal 3D backbone:
      - All 3x3x3 convs with BN+ReLU
      - Pool1 keeps time length (1x2x2), others downsample time too (2x2x2)
      - Global 3D avg pool + small MLP head
    Expected input: [B, 3, T, H, W]
    """
    def __init__(self, num_classes=10, t_frames=10):
        super().__init__()
        pool5 = (1,2,2) if t_frames <= 10 else (2,2,2)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(3, 32),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),   # Pool1: keep T

            block(32, 64),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),   # Pool2

            block(64, 128),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),   # Pool3

            block(128, 256),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),   # Pool4

            block(256, 512),
            nn.MaxPool3d(kernel_size=pool5, stride=pool5),       # Pool5 (safe for short clips)
        )

        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):           # x: [B, 3, T, H, W]
        x = self.features(x)
        x = self.gap(x)             # [B, 512, 1,1,1]
        x = self.head(x)            # [B, num_classes]
        return x

# Peek T to configure Pool5 safely
with torch.no_grad():
    sample_x, _ = next(iter(train_loader))  # [B, C, T, H, W]
    T_frames = sample_x.shape[2]

model = SimpleC3D(num_classes=num_classes, t_frames=T_frames).to(device)

# ---------------- OPTIMIZER & LOSS ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ---------------- TRAIN / EVAL HELPERS ----------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_correct, total_loss, total = 0, 0.0, 0
    for data, target in loader:
        # temporal center crop BEFORE sending to device
        data = temporal_center_crop(data, t_target=T_TARGET)

        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * target.size(0)
        total_correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)
    return total_loss/total, total_correct/total


# ---------------- TRAINING LOOP ----------------
best_val = 0.0
for epoch in range(num_epochs):
    model.train()
    train_correct, train_loss_sum, seen = 0, 0.0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for data, target in pbar:
        # === NEW: random temporal crop/jitter for training ===
        data = temporal_jitter(data, t_target=T_TARGET, max_stride=2, reverse_p=0.1)

        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * target.size(0)
        train_correct  += (output.argmax(1) == target).sum().item()
        seen += target.size(0)

    train_loss = train_loss_sum / seen
    train_acc  = train_correct / seen
    val_loss, val_acc = evaluate(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", train_acc, epoch)
    writer.add_scalar("Val/Loss", val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "model_best_c3d_simple.pth")
        print("✓ Saved: model_best_c3d_simple.pth")


writer.close()
print("✅ Training finished.")

# ---------------- TEST ----------------
state = torch.load("model_best_c3d_simple.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
test_loss, test_acc = evaluate(test_loader)
print(f"🎬 Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

# ---------------- SAVE METRICS ----------------
results = {
    "model": "SimpleC3D(3D VGG-style)",
    "epochs": num_epochs,
    "val_best_acc": best_val,
    "test_acc": test_acc,
    "img_size": img_size,
    "batch_size": batch_size
}
with open("train_results_c3d_simple.json", "w") as f:
    json.dump(results, f, indent=4)
print("✅ Results saved to train_results_c3d_simple.json")
