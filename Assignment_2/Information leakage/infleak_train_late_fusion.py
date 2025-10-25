# train_late_fusion.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, json, math
from collections import defaultdict
from typing import List
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import FrameVideoDataset
from single_frame_CNN import Network   # <-- re-use your CNN
from tqdm import tqdm

# ---------------- CONFIG ----------------
root_dir = '/dtu/datasets1/02516/ucf101_noleakage'
num_classes = 10
T_FRAMES = 10                # your dataset has 10 frames/clip
BATCH_SIZE = 4               # we’ll batch videos (see collate below)
EPOCHS = 8
LR = 1e-3
WD = 1e-4
USE_HEAD = True             # True = learnable fusion head; False = mean logits
LOAD_SINGLE_WEIGHTS = "model_best_single_frame.pth"  # state_dict from your single-frame training
SAVE_AS = "late_fusion.pt"
LOG_JSON = "train_results_late.json"

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- DATA ----------------
base_tf = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize([0.45]*3, [0.225]*3),
])

trainset = FrameVideoDataset(root_dir, split="train", transform=base_tf, stack_frames=False)
valset   = FrameVideoDataset(root_dir, split="val",   transform=base_tf, stack_frames=False)
testset  = FrameVideoDataset(root_dir, split="test",  transform=base_tf, stack_frames=False)

def collate_video(batch):
    """
    Input batch: list of items, each item is (frames_list_or_tensor, label).
    For stack_frames=False, dataset returns a Python list of 10 frame tensors [C,H,W].
    We will stack per-video frames into a tensor [T, C, H, W], then batch into [B, T, C, H, W].
    """
    vids, labels = [], []
    for frames, y in batch:
        # frames: list of 10 tensors [3,64,64]
        if isinstance(frames, list):
            frames = torch.stack(frames, dim=0)  # [T, C, H, W]
        vids.append(frames)
        labels.append(y)
    # pad not needed if all videos have same T (10)
    vids = torch.stack(vids, dim=0)             # [B, T, C, H, W]
    labels = torch.tensor(labels, dtype=torch.long)
    return vids, labels

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_video)
val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_video)
test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_video)

# ---------------- MODEL ----------------
# Backbone = your single-frame classifier (per-frame logits)
backbone = Network(num_classes=num_classes).to(device)
# load state_dict saved by your single_frame_CNN.py
state = torch.load(LOAD_SINGLE_WEIGHTS, map_location=device, weights_only=True)
backbone.load_state_dict(state)

# (Optional) freeze backbone for quick training:
# for p in backbone.parameters(): p.requires_grad = False

# Late fusion head (optional)
if USE_HEAD:
    fuse_head = nn.Linear(num_classes, num_classes).to(device)
    params = list(backbone.parameters()) + list(fuse_head.parameters())
else:
    fuse_head = None
    params = list(backbone.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params, lr=LR, weight_decay=WD)

# ---------------- TRAIN / EVAL LOOPS ----------------
def forward_video_batch(vidsBTCHW):
    """
    vidsBTCHW: [B, T, C, H, W]
    Run per-frame through backbone to get logits [B,T,num_classes],
    then fuse across T (mean or head(mean)).
    """
    B, T, C, H, W = vidsBTCHW.shape
    vidsBTC = vidsBTCHW.view(B*T, C, H, W)             # flatten frames
    logitsBT = backbone(vidsBTC)                       # [B*T, num_classes]
    logitsBT = logitsBT.view(B, T, num_classes)        # [B, T, num_classes]
    mean_logits = logitsBT.mean(dim=1)                 # [B, num_classes]
    if fuse_head is not None:
        mean_logits = fuse_head(mean_logits)           # [B, num_classes]
    return mean_logits

def run_epoch(loader, train=True):
    model_parts = [backbone] + ([fuse_head] if fuse_head is not None else [])
    for m in model_parts:
        m.train(mode=train)

    total_loss, correct, total = 0.0, 0, 0
    iter_loader = tqdm(loader, desc="Train" if train else "Eval", leave=False)
    for vids, labels in iter_loader:
        vids, labels = vids.to(device), labels.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        logits = forward_video_batch(vids)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss/total, correct/total

best_val = 0.0
history = []
for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    print(f"Epoch {ep}/{EPOCHS}  Train {tr_acc*100:.2f}% (loss {tr_loss:.4f})  Val {va_acc*100:.2f}% (loss {va_loss:.4f})")
    history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": va_acc, "train_loss": tr_loss, "val_loss": va_loss})
    if va_acc > best_val:
        best_val = va_acc
        torch.save({
            "backbone": backbone.state_dict(),
            "fuse_head": (fuse_head.state_dict() if fuse_head is not None else None),
            "use_head": USE_HEAD,
            "num_classes": num_classes
        }, SAVE_AS)
        print("✓ Saved:", SAVE_AS)

# Final test
ckpt = torch.load(SAVE_AS, map_location=device, weights_only=False)
backbone.load_state_dict(ckpt["backbone"])
if ckpt["use_head"]:
    if fuse_head is None:
        fuse_head = nn.Linear(num_classes, num_classes).to(device)
    fuse_head.load_state_dict(ckpt["fuse_head"])

te_loss, te_acc = run_epoch(test_loader, train=False)
print(f"Test Accuracy (video-level): {te_acc*100:.2f}%")

with open(LOG_JSON, "w") as f:
    json.dump({
        "model": "LateFusion(mean logits)" if not USE_HEAD else "LateFusion(mean logits + head)",
        "epochs": EPOCHS,
        "best_val_acc": best_val,
        "test_acc": te_acc,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WD,
        "use_head": USE_HEAD
    }, f, indent=4)
print("Saved metrics to", LOG_JSON)
