# eval_confusions.py
import os, json, math, argparse, collections
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets import FrameVideoDataset
from single_frame_CNN import Network  # backbone (32,64,128,256,512)

# ---------- Config ----------
root_dir = '/dtu/datasets1/02516/ufc10'
IMG_SIZE = 64
T_FRAMES = 10
BATCH_VID = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- Transforms ----------
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.45,0.45,0.45],[0.225,0.225,0.225]),
])

# ---------- Helpers ----------
def get_class_names_from_dataset(ds, num_classes=10):
    # Try common attributes/fields; fallback to numeric strings
    # Aim: idx -> label name
    # 1) classes attr
    if hasattr(ds, 'classes'):
        return list(ds.classes)
    # 2) df with 'label' and a string column (e.g. 'class'/'action')
    try:
        df = ds.df  # pandas DataFrame, if present
        if 'label' in df.columns:
            # find a likely name column
            for cand in ['class', 'action', 'label_name', 'cls', 'category']:
                if cand in df.columns:
                    # Build by majority name per idx to be safe
                    lab2name = {}
                    for k, g in df.groupby('label'):
                        names = g[cand].astype(str).values
                        if len(names):
                            # pick the most frequent
                            name = collections.Counter(names).most_common(1)[0][0]
                            lab2name[int(k)] = name
                    # make dense list
                    mx = max(lab2name.keys())
                    return [lab2name.get(i, str(i)) for i in range(mx+1)]
    except Exception:
        pass
    # 3) fallback
    return [str(i) for i in range(num_classes)]

def majority_vote(preds_1d):
    # preds_1d: list/np of ints
    return collections.Counter(preds_1d).most_common(1)[0][0]

def save_confusion(cm, class_names, prefix):
    cm = np.asarray(cm, dtype=np.int64)
    # CSV
    import csv
    with open(f'confusion_{prefix}.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['']+class_names)
        for i,row in enumerate(cm):
            w.writerow([class_names[i]]+list(row))
    print(f"Saved confusion_{prefix}.csv")

    # PNG heatmap
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix - {prefix}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(f'confusion_{prefix}.png', dpi=160)
    plt.close()
    print(f"Saved confusion_{prefix}.png")

# ---------- Single-frame (video-level via majority vote) ----------
def eval_singleframe_majority():
    ds = FrameVideoDataset(root_dir, split='test', transform=tf, stack_frames=False)
    class_names = get_class_names_from_dataset(ds)
    num_classes = len(class_names)

    # model
    model = Network(num_classes=num_classes).to(device)
    state = torch.load("model_best_single_frame.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # loader: batch videos (list-of-frames), collate to tensor [B,T,C,H,W]
    def collate_video(batch):
        vids, labels = [], []
        for frames, y in batch:
            # frames: list of T tensors [C,H,W]
            vids.append(torch.stack(frames, dim=0))  # [T,C,H,W]
            labels.append(y)
        return torch.stack(vids, dim=0), torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(ds, batch_size=BATCH_VID, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_video)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for vids, labels in loader:
            vids, labels = vids.to(device), labels.to(device)  # vids: [B,T,C,H,W]
            B,T,C,H,W = vids.shape
            # flatten frames -> [B*T,C,H,W]
            framesBT = vids.view(B*T, C, H, W)
            logitsBT = model(framesBT)  # [B*T, num_classes]
            predsBT = torch.argmax(logitsBT, dim=1).view(B, T)  # per-video frame preds

            # majority over T
            for i in range(B):
                mv = majority_vote(predsBT[i].tolist())
                cm[labels[i].item(), mv] += 1

    save_confusion(cm, class_names, prefix='single')

# ---------- Late fusion (mean logits) ----------
def eval_latefusion():
    # We expect checkpoint with keys: backbone, fuse_head, use_head
    ckpt = torch.load("late_fusion.pt", map_location=device, weights_only=False)
    use_head = bool(ckpt.get("use_head", False))

    # dataset
    ds = FrameVideoDataset(root_dir, split='test', transform=tf, stack_frames=False)
    class_names = get_class_names_from_dataset(ds)
    num_classes = len(class_names)

    # models
    backbone = Network(num_classes=num_classes).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    fuse_head = None
    if use_head:
        fuse_head = nn.Linear(num_classes, num_classes).to(device)
        fuse_head.load_state_dict(ckpt["fuse_head"])

    backbone.eval()
    if fuse_head is not None: fuse_head.eval()

    # loader: [B,T,C,H,W]
    def collate_video(batch):
        vids, labels = [], []
        for frames, y in batch:
            vids.append(torch.stack(frames, dim=0))  # [T,C,H,W]
            labels.append(y)
        return torch.stack(vids, dim=0), torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(ds, batch_size=BATCH_VID, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_video)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for vids, labels in loader:
            vids, labels = vids.to(device), labels.to(device)
            B,T,C,H,W = vids.shape
            # per-frame logits then mean
            framesBT = vids.view(B*T, C, H, W)
            logitsBT = backbone(framesBT).view(B, T, num_classes)  # [B,T,C]
            mean_logits = logitsBT.mean(dim=1)                     # [B,C]
            if fuse_head is not None:
                mean_logits = fuse_head(mean_logits)
            preds = torch.argmax(mean_logits, dim=1)               # [B]
            for i in range(B):
                cm[labels[i].item(), preds[i].item()] += 1

    save_confusion(cm, class_names, prefix='late')

# ---------- Early fusion (channels = 3*T) ----------
class Early2D(nn.Module):
    # MUST mirror what you trained (32,64,128,256,512), and forward with 3T-channels trick
    def __init__(self, T=10, num_classes=10):
        super().__init__()
        in_ch = 3*T
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),     nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1),     nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3, padding=1),     nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256,512,3, padding=1),     nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):  # x: [B,3,T,H,W]
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B, C*T, H, W)  # [B, 3T, H, W]
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

def eval_earlyfusion():
    ds = FrameVideoDataset(root_dir, split='test', transform=tf, stack_frames=True)  # -> [B,3,T,H,W]
    class_names = get_class_names_from_dataset(ds)
    num_classes = len(class_names)

    model = Early2D(T=T_FRAMES, num_classes=num_classes).to(device)
    state = torch.load("early_fusion.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(ds, batch_size=BATCH_VID, shuffle=False, num_workers=4, pin_memory=True)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:      # x: [B,3,T,H,W]
            x, y = x.to(device), y.to(device)
            logits = model(x)    # [B,num_classes]
            preds = torch.argmax(logits, dim=1)
            for i in range(x.size(0)):
                cm[y[i].item(), preds[i].item()] += 1

    save_confusion(cm, class_names, prefix='early')

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, default="all", choices=["all","single","late","early"])
    args = parser.parse_args()

    if args.which in ("all","single"):
        print("\n=== Single-frame (majority vote) ===")
        eval_singleframe_majority()
    if args.which in ("all","late"):
        print("\n=== Late fusion (mean logits) ===")
        eval_latefusion()
    if args.which in ("all","early"):
        print("\n=== Early fusion (3T channels) ===")
        eval_earlyfusion()

    print("\nDone.")
