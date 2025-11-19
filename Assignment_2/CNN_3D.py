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

trainset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames = True)
valset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
testset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames = True)


train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
val_loader   = DataLoader(valset, batch_size=8, shuffle=False)
test_loader  = DataLoader(testset, batch_size=8, shuffle=False)


# ---------------- MODEL ----------------


class Network_3D(nn.Module):
    def __init__(self, num_classes=10):
        super(Network_3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # ← temporal stride removed here

            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # ← same here

            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # ← only spatial pooling
        )


        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # global average over time + space
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, T, H, W]
        x = self.features(x)
        x = self.gap(x)             # [B, 512, 1, 1, 1]
        x = torch.flatten(x, 1)     # [B, 512]
        x = self.fc(x)              # [B, num_classes]
        return x

model = Network_3D(num_classes=10).to(device)

# ---------------- OPTIMIZER & LOSS ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ---------------- TRAINING LOOP ----------------
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_loss = 0.0

    # TRAINING
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Metrics
        train_loss += loss.item() * data.size(0)
        predicted = output.argmax(1)
        train_correct += (predicted == target).sum().item()

    # ---------------- EVALUATION ----------------
    model.eval()
    test_correct = 0
    test_loss = 0.0
    incorrect_indices = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            probs = torch.softmax(output, dim=1)
            predicted = probs.argmax(1)
            test_correct += (target == predicted).sum().item()

            incorrect = (predicted != target).nonzero(as_tuple=True)[0]
            for i in incorrect:
                incorrect_indices.append(batch_idx * test_loader.batch_size + i.item())

    # ---------------- METRICS ----------------
    train_acc = train_correct / len(trainset)
    test_acc = test_correct / len(testset)
    train_loss /= len(trainset)
    test_loss /= len(testset)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")

print("✅ Training finished.")

torch.save(model, "Assignment_2/model_3D_best.pth")

print("✅ Model saved.")
