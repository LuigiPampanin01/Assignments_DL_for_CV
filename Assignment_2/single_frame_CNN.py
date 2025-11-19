from datasets import FrameImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # ✅ safe for HPC/headless jobs
import matplotlib.pyplot as plt


# ---------------- MODEL ----------------
class Network(nn.Module):
    def __init__(self, num_classes=10):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
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


# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
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
    # root_dir = '/dtu/datasets1/02516/ucf101_noleakage'
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    train_transform = T.Compose([
        T.Resize((72, 72)),
        T.RandomCrop((64, 64)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    trainset = FrameImageDataset(root_dir=root_dir, split='train', transform=train_transform)
    valset   = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    testset  = FrameImageDataset(root_dir=root_dir, split='test', transform=transform)

    train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(valset, batch_size=8, shuffle=False)
    test_loader  = DataLoader(testset, batch_size=8, shuffle=False)

    # ---------------- MODEL SETUP ----------------
    model = Network(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ---------------- TRACKING METRICS ----------------
    train_acc_history, test_acc_history = [], []
    train_loss_history, test_loss_history = [], []

    # ---------------- TRAINING LOOP ----------------
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss = 0.0

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            predicted = output.argmax(1)
            train_correct += (predicted == target).sum().item()

        # ---------------- EVALUATION ----------------
        model.eval()
        test_correct = 0
        test_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                predicted = output.argmax(1)
                test_correct += (predicted == target).sum().item()

        # ---------------- METRICS ----------------
        train_acc = train_correct / len(trainset)
        test_acc = test_correct / len(testset)
        train_loss /= len(trainset)
        test_loss /= len(testset)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")

    print("✅ Training finished.")

    # ---------------- PLOTS ----------------
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), train_acc_history, label='Train Accuracy', marker='o')
    plt.plot(range(1, num_epochs+1), test_acc_history, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Assignment_2/accuracy_plot.png")
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), train_loss_history, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs+1), test_loss_history, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Assignment_2/loss_plot.png")
    plt.close()

    print("📊 Saved accuracy_plot.png and loss_plot.png to Assignment_2/")

    # save model weights only (safer for reloading)
    torch.save(model.state_dict(), "Assignment_2/model_best_single_frame.pth")
    print("✅ Model weights saved.")
