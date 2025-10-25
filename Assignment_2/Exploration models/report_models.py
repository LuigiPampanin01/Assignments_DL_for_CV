# train_late_fusion.py
import os, json, torch, argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from datasets import FrameVideoDataset
from single_frame_CNN import Network  # must define forward_features()

# ---------------- CONFIG ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ufc10", choices=["ufc10", "ucf101_noleakage"])
parser.add_argument("--fusion", type=str, default="mean", choices=["mean", "fc"])
args = parser.parse_args()

# Consistent hyperparams with teammates
num_classes = 10
T_FRAMES = 10
BATCH_SIZE = 8
EPOCHS = 24
LR = 1e-3
WD = 1e-4
LOAD_SINGLE_WEIGHTS = "model_best_single_frame.pth"

root_dir = f"/dtu/datasets1/02516/{args.dataset}"
SAVE_AS = f"latefusion_{args.fusion}_{args.dataset}.pt"
LOG_JSON = f"latefusion_{args.fusion}_{args.dataset}.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} | Dataset: {args.dataset} | Fusion: {args.fusion}")

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
    vids, labels = [], []
    for frames, y in batch:
        if isinstance(frames, list):
            frames = torch.stack(frames, dim=0)
        vids.append(frames)
        labels.append(y)
    vids = torch.stack(vids, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return vids, labels

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_video)
val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_video)
test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_video)

# ---------------- MODEL ----------------
backbone = Network(num_classes=num_classes).to(device)
state = torch.load(LOAD_SINGLE_WEIGHTS, map_location=device, weights_only=True)
backbone.load_state_dict(state)

# Utility: ensure backbone has forward_features()
if not hasattr(backbone, "forward_features"):
    raise AttributeError("Your Network must define a forward_features(x) returning penultimate feature vector.")

# Fusion choice
if args.fusion == "mean":
    fusion_head = nn.Linear(num_classes, num_classes).to(device)
    def forward_video_batch(vidsBTCHW):
        B, T, C, H, W = vidsBTCHW.shape
        vidsBTC = vidsBTCHW.view(B*T, C, H, W)
        logitsBT = backbone(vidsBTC)            # [B*T, num_classes]
        logitsBT = logitsBT.view(B, T, num_classes)
        mean_logits = logitsBT.mean(dim=1)
        return fusion_head(mean_logits)

else:  # args.fusion == "fc"
    # determine feature size dynamically
    with torch.no_grad():
        dummy = torch.randn(1, 3, 64, 64).to(device)
        feat_dim = backbone.forward_features(dummy).numel()

    fusion_head = nn.Sequential(
        nn.Linear(T_FRAMES * feat_dim, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    ).to(device)

    def forward_video_batch(vidsBTCHW):
        B, T, C, H, W = vidsBTCHW.shape
        vidsBTC = vidsBTCHW.view(B*T, C, H, W)
        featsBT = backbone.forward_features(vidsBTC)  # [B*T, D]
        D = featsBT.shape[-1]
        featsBT = featsBT.view(B, T*D)
        logits = fusion_head(featsBT)
        return logits

params = list(backbone.parameters()) + list(fusion_head.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params, lr=LR, weight_decay=WD)

# ---------------- TRAIN / EVAL ----------------
def run_epoch(loader, train=True):
    backbone.train(mode=train)
    fusion_head.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    desc = "Train" if train else "Eval"
    for vids, labels in tqdm(loader, desc=desc, leave=False):
        vids, labels = vids.to(device), labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = forward_video_batch(vids)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

best_val = 0.0
history = []
for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)
    print(f"Epoch {ep}/{EPOCHS} | Train {tr_acc*100:.2f}% (loss {tr_loss:.3f}) | Val {va_acc*100:.2f}% (loss {va_loss:.3f})")
    history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": va_acc})
    if va_acc > best_val:
        best_val = va_acc
        torch.save({
            "backbone": backbone.state_dict(),
            "fusion_head": fusion_head.state_dict(),
            "fusion": args.fusion,
            "num_classes": num_classes
        }, SAVE_AS)
        print("✓ Saved new best:", SAVE_AS)

# ---------------- FINAL TEST ----------------
ckpt = torch.load(SAVE_AS, map_location=device)
backbone.load_state_dict(ckpt["backbone"])
fusion_head.load_state_dict(ckpt["fusion_head"])

test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ---------------- LOG METRICS ----------------
with open(LOG_JSON, "w") as f:
    json.dump({
        "fusion": args.fusion,
        "dataset": args.dataset,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WD,
        "best_val_acc": best_val,
        "test_acc": test_acc
    }, f, indent=4)
print("Saved metrics to", LOG_JSON)
