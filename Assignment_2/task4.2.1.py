from data import FrameImageDatasetTwoStream
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
root_dir = '/dtu/datasets1/02516/ucf101_noleakage'

# Transform for validation/test (no augmentation)
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# Transform for training (with augmentation)
train_transform = T.Compose([
    T.Resize((72, 72)),
    T.RandomCrop((64, 64)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# Transform for flow (grayscale, no color augmentation)
flow_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45], std=[0.225, 0.225])
])

trainset = FrameImageDatasetTwoStream(root_dir=root_dir, split='train', transform=transform, transform2=flow_transform)
valset = FrameImageDatasetTwoStream(root_dir=root_dir, split='val', transform=transform, transform2=flow_transform)
testset = FrameImageDatasetTwoStream(root_dir=root_dir, split='test', transform=transform, transform2=flow_transform)

train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
val_loader = DataLoader(valset, batch_size=8, shuffle=False)
test_loader = DataLoader(testset, batch_size=8, shuffle=False)

print(f"Train samples: {len(trainset)}")
print(f"Val samples: {len(valset)}")
print(f"Test samples: {len(testset)}")

# ---------------- MODEL ----------------
class Network(nn.Module):
    def __init__(self, num_classes, batch_number, RGB):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.RGB = RGB
        self.batch_number = batch_number
        self.features = nn.Sequential(
            nn.Conv2d(RGB * batch_number, 32, 3, padding=1),
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
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# RGB model: 1 frame, 3 channels
model_rgb = Network(num_classes=10, batch_number=1, RGB=3).to(device)
# Flow model: 9 frames, 2 channels per frame (x and y flow)
model_flow = Network(num_classes=10, batch_number=9, RGB=2).to(device)

# ---------------- OPTIMIZER & LOSS ----------------
criterion = nn.CrossEntropyLoss()

# Learnable fusion weight
alpha = nn.Parameter(torch.tensor(0.8, device=device))

# Combine parameters from both models + alpha
params = list(model_rgb.parameters()) + list(model_flow.parameters()) + [alpha]
optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

# Loss weight for auxiliary losses
lam = 0.3
num_epochs = 10

# ---------------- TRAINING LOOP ----------------
for epoch in range(num_epochs):
    model_rgb.train()
    model_flow.train()
    train_correct = 0
    train_loss_sum = 0.0
    train_seen = 0

    # TRAINING
    for data_rgb, target, data_flow in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        # data_rgb: [B, 3, H, W]
        # data_flow: [B, 9, 2, H, W]
        # target: [B]
        
        data_rgb = data_rgb.to(device)
        target = target.to(device)
        data_flow = data_flow.to(device)

        B, T, C, H, W = data_flow.shape

        data_flow = data_flow.view(B, T * 2, H, W)
        
        optimizer.zero_grad()

        # Forward pass
        rgb_logits = model_rgb(data_rgb)
        flow_logits = model_flow(data_flow)
        
        # Fused prediction with learnable alpha
        fused_logits = alpha * rgb_logits + (1 - alpha) * flow_logits
        
        # Loss: fused + auxiliary losses
        loss = criterion(fused_logits, target) \
             + lam * criterion(rgb_logits, target) \
             + lam * criterion(flow_logits, target)
        
        loss.backward()
        optimizer.step()

        # Metrics
        train_loss_sum += loss.item() * data_rgb.size(0)
        predicted = fused_logits.argmax(1)
        train_correct += (predicted == target).sum().item()
        train_seen += data_rgb.size(0)

    model_rgb.eval()
    model_flow.eval()
    test_correct = 0
    test_loss_sum = 0.0
    test_seen = 0

    with torch.no_grad():
        for data_rgb, target, data_flow in tqdm(test_loader, desc="Testing"):
            data_rgb = data_rgb.to(device)
            target = target.to(device)
            data_flow = data_flow.to(device)
            
            # Reshape flow
            B, T, C, H, W = data_flow.shape
            data_flow = data_flow.view(B, T * C, H, W)

            # Forward pass
            rgb_logits = model_rgb(data_rgb)
            flow_logits = model_flow(data_flow)
            fused_logits = alpha * rgb_logits + (1 - alpha) * flow_logits

            # Loss
            loss = criterion(fused_logits, target) \
                + lam * criterion(rgb_logits, target) \
                + lam * criterion(flow_logits, target)
            
            # Metrics
            predicted = fused_logits.argmax(1)
            test_correct += (predicted == target).sum().item()
            test_loss_sum += loss.item() * data_rgb.size(0)
            test_seen += data_rgb.size(0)

    train_acc = train_correct / len(trainset)
    test_acc = test_correct / len(testset)
    train_loss = train_loss_sum / len(trainset)
    test_loss = test_loss_sum / len(testset)


    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")




# ---------------- SAVE MODEL ----------------
torch.save({
    'model_rgb': model_rgb.state_dict(),
    'model_flow': model_flow.state_dict(),
    'alpha': alpha.item(),
}, "model4_2_2.pth")

print("✅ Model saved to model4_2_2.pth")