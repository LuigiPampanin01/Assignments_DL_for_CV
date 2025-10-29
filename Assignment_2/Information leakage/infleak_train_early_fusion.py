# train_early_fusion.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, json, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from datasets import FrameVideoDataset

# ---------------- CONFIG ----------------
root_dir    = '/dtu/datasets1/02516/ucf101_noleakage'
num_classes = 10
EPOCHS      = 20
BATCH       = 32
LR          = 3e-4
WD          = 1e-4
SAVE_AS     = "early_fusion.pt"
LOG_JSON    = "train_results_early.json"
IMG_SIZE    = 64
T_FRAMES    = 10

# ---------------- SEED ----------------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- DATA ----------------
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.45,0.45,0.45], [0.225,0.225,0.225]),
])

# We want stacked frames: x ∈ [3, T, H, W]
trainset = FrameVideoDataset(root_dir, "train", tf, stack_frames=True)
valset   = FrameVideoDataset(root_dir, "val",   tf, stack_frames=True)
testset  = FrameVideoDataset(root_dir, "test",  tf, stack_frames=True)

train_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(valset,   batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(testset,  batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

# ---------------- MODEL: Early Fusion 2D (channels = 3*T) ----------------
# Match single-frame backbone widths exactly: [32, 64, 128, 256, 512]
class Early2D(nn.Module):
    def __init__(self, T=10, num_classes=10):
        super().__init__()
        in_ch = 3 * T
        self.features = nn.Sequential(
            nn.Conv2d(in_ch,  32, 3, padding=1),  nn.BatchNorm2d( 32), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 32 x 32
            nn.Conv2d( 32,   64, 3, padding=1),  nn.BatchNorm2d( 64), nn.ReLU(), nn.MaxPool2d(2),  # 64 x 16 x 16
            nn.Conv2d( 64,  128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 128 x 8 x 8
            nn.Conv2d(128,  256, 3, padding=1),  nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),  # 256 x 4 x 4
            nn.Conv2d(256,  512, 3, padding=1),  nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),  # 512 x 2 x 2
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,num_classes))

    def forward(self, x):  # x: [B, 3, T, H, W]
        B, C, T, H, W = x.shape
        # Early fusion: stack frames along channels → [B, 3*T, H, W]
        x = x.permute(0,2,1,3,4).reshape(B, C*T, H, W)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

model = Early2D(T=T_FRAMES, num_classes=num_classes).to(device)

# ---- Initialize Early2D from single-frame weights (safe selective load) ----
def safe_load_singleframe_into_early(model, ckpt_path, T=10):
    sf = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # 1) Copy matching-shaped layers (skip conv1 by hand)
    msd = model.state_dict()
    filtered = {}
    for k, v in sf.items():
        if k == "features.0.weight" or k == "features.0.bias":
            continue  # handle conv1 separately
        if k in msd and msd[k].shape == v.shape:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # 2) Inflate conv1 from [32,3,3,3] -> [32, 3*T, 3, 3]
    with torch.no_grad():
        w3 = sf["features.0.weight"]          # [32, 3, 3, 3]
        b  = sf["features.0.bias"]            # [32]
        w30 = w3.repeat(1, T, 1, 1) / T       # tile along channel dim, average to keep scale
        model.features[0].weight.copy_(w30)
        model.features[0].bias.copy_(b)
    print("✓ Early2D: loaded matching layers + inflated conv1 from single-frame.")

try:
    safe_load_singleframe_into_early(model, "model_best_single_frame.pth", T=T_FRAMES)
except Exception as e:
    print("Init from single-frame skipped:", e)


# Loss, Optim, Scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------- TRAIN / EVAL ----------------
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    it = tqdm(loader, desc="Train" if train else "Eval", leave=False)
    for x, y in it:
        x, y = x.to(device), y.to(device)        # x:[B,3,T,H,W], y:[B]
        if train: optimizer.zero_grad(set_to_none=True)
        out = model(x)                           # [B, num_classes]
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

best_val = 0.0
history = []
for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader,   False)
    print(f"Epoch {ep}/{EPOCHS}  Train {tr_acc*100:.2f}% (loss {tr_loss:.4f})  Val {va_acc*100:.2f}% (loss {va_loss:.4f})")
    history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": va_acc, "train_loss": tr_loss, "val_loss": va_loss})
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), SAVE_AS)
        print("✓ Saved:", SAVE_AS)
    scheduler.step()

# ---------------- TEST ----------------
state = torch.load(SAVE_AS, map_location=device, weights_only=True)
model.load_state_dict(state)
te_loss, te_acc = run_epoch(test_loader, False)
print(f"Test Accuracy (video-level): {te_acc*100:.2f}%")

# ---------------- LOG ----------------
with open(LOG_JSON, "w") as f:
    json.dump({
        "model": "Early2D(3*T channels, conv1-inflated)",
        "epochs": EPOCHS,
        "best_val_acc": best_val,
        "test_acc": te_acc,
        "batch": BATCH,
        "lr": LR,
        "wd": WD,
        "img_size": IMG_SIZE,
        "T": T_FRAMES,
        "seed": SEED
    }, f, indent=4)
print("Saved metrics to", LOG_JSON)
