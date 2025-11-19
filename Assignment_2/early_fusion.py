from datasets import FrameVideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

# ---------------- DEVICE SETUP ----------------
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
root_dir = 'Assignment_2/ufc10'
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

train_transform = T.Compose([
    T.Resize((72, 72)),  # resize slightly larger for cropping
    T.RandomCrop((64, 64)),  # random crop for spatial variation
    T.RandomHorizontalFlip(p=0.5),  # mirror frames
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # lighting/color variation
    T.RandomRotation(degrees=10),  # small random rotations
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

trainset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames = False)
valset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)
testset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames = False)


train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
val_loader   = DataLoader(valset, batch_size=8, shuffle=False)
test_loader  = DataLoader(testset, batch_size=8, shuffle=False)

# inspection 

batch_1 = next(iter(train_loader))
frames_batch, labels_batch = batch_1

T = len(frames_batch)                # number of frames per video
B = frames_batch[0].shape[0]         # number of videos in the batch
C, H, W = frames_batch[0].shape[1:]  # frame dimensions

print(f"Batch size (B): {B}")
print(f"Frames per video (T): {T}")
print(f"Frame shape: [{C}, {H}, {W}]")
print(f"Labels shape: {labels_batch.shape}")

class EarlyFusionNetwork(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super(EarlyFusionNetwork, self).__init__()
        
        self.num_frames = num_frames

        self.features = nn.Sequential(
            nn.Conv2d(3 * num_frames, 32, 3, padding=1),   # 3*T input channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # 🔹 Fuse frames early: stack RGB channels across time
        x = x.view(B, T * C, H, W)   # → [B, 3*T, H, W]

        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

model = EarlyFusionNetwork(num_classes=10).to(device)

# ---------------- OPTIMIZER & LOSS ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 20

best_acc = 0.0

for epoch in range(num_epochs):
    # ---------- TRAIN ----------
    model.train()
    train_loss, train_correct = 0.0, 0

    for frames, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        frames = torch.stack(frames, dim=1).to(device)   # [B,T,C,H,W]
        target = target.to(device)

        optimizer.zero_grad()
        output = model(frames)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * frames.size(0)
        preds = output.argmax(1)
        train_correct += (preds == target).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # ---------- TEST DURING TRAINING ----------
    model.eval()
    test_loss, test_correct = 0.0, 0

    with torch.no_grad():
        for frames, target in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test Eval]"):
            frames = torch.stack(frames, dim=1).to(device)
            target = target.to(device)

            output = model(frames)
            loss = criterion(output, target)

            test_loss += loss.item() * frames.size(0)
            preds = output.argmax(1)
            test_correct += (preds == target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)

    # save best model based on test performance
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "Assignment_2/model_best_early_fusion.pth")

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% || "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")

print("\n✅ Training finished.")
print(f"Best test accuracy achieved during training: {best_acc*100:.2f}%")

# ---------- FINAL VALIDATION EVALUATION ----------
print("\n🔍 Evaluating best model on VALIDATION set...")
model.load_state_dict(torch.load("Assignment_2/model_best_early_fusion.pth"))
model.eval()
val_loss, val_correct = 0.0, 0
incorrect_indices = []

with torch.no_grad():
    for batch_idx, (frames, target) in enumerate(tqdm(val_loader, desc="Validation Eval")):
        frames = torch.stack(frames, dim=1).to(device)
        target = target.to(device)

        output = model(frames)
        loss = criterion(output, target)
        val_loss += loss.item() * frames.size(0)

        preds = output.argmax(1)
        val_correct += (preds == target).sum().item()

        # store incorrect samples (optional)
        incorrect = (preds != target).nonzero(as_tuple=True)[0]
        for i in incorrect:
            incorrect_indices.append(batch_idx * val_loader.batch_size + i.item())

val_loss /= len(val_loader.dataset)
val_acc = val_correct / len(val_loader.dataset)

print(f"\n✅ Final Validation Results → Loss: {val_loss:.4f} | Accuracy: {val_acc*100:.1f}%")

