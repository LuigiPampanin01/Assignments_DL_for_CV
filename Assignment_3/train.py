import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from utils.dataclass_potholes import PotholeDataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from model.model import ResNetPatchClassifier
from utils.losses import FocalLoss

# Dataset
def create_balanced_sampler(dataset):
    all_labels = []

    # Check if the input is a PyTorch Subset object
    if isinstance(dataset, Subset):
        # If it's a Subset, get the original dataset and the indices list
        original_dataset = dataset.dataset
        subset_indices = dataset.indices
        
        # Iterate over the indices used to create the subset
        for idx in subset_indices:
            # The indices are applied to the original dataset's indexing
            # We assume original_dataset.index is a list/tuple of (sample_index, patch_index)
            # where the length of original_dataset.index is len(original_dataset)
            
            # Get the sample/patch indices (si, pi) for this specific data point
            si, pi = original_dataset.index[idx] 
            
            # Use these indices to fetch the label
            all_labels.append(original_dataset.samples[si].labels[pi].item())
            
    else:
        # If it's the original PotholeDataset, use the existing logic
        # Collect labels in order of dataset.index
        for si, pi in dataset.index:
            all_labels.append(dataset.samples[si].labels[pi].item())

    # --- Rest of the function remains the same ---
    labels_tensor = torch.tensor(all_labels)

    pos_count = (labels_tensor == 1).sum().item()
    neg_count = (labels_tensor == 0).sum().item()

    # sampling weights
    weight_pos = neg_count / pos_count
    weight_neg = 1.0

    weights = torch.where(
        labels_tensor == 1,
        torch.tensor(weight_pos, dtype=torch.float),
        torch.tensor(weight_neg, dtype=torch.float)
    )

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.08),
    transforms.ToTensor(),
])

# ----------------------------------------
# Load TRAIN SET
# ----------------------------------------
train_set = PotholeDataset(proposal_dir="utils/dataset_proposal",image_dir="/dtu/datasets1/02516/potholes/images",transform=transform,
    split=(70, 20, 10),
    mode="train"
)
test_set = PotholeDataset(proposal_dir="utils/dataset_proposal",image_dir="/dtu/datasets1/02516/potholes/images",transform=transform_test,
    split=(70, 20, 10),
    mode="test"
)
eval_set = PotholeDataset(proposal_dir="utils/dataset_proposal",image_dir="/dtu/datasets1/02516/potholes/images",transform=transform_test,
    split=(70, 20, 10),
    mode="eval"
)

indices = list(range(0,80))
subset_train = Subset(train_set, indices)
subset_test = Subset(test_set, indices)
subset_eval = Subset(eval_set, indices)
sampler = create_balanced_sampler(train_set)
sampler_subset = create_balanced_sampler(subset_train)

train_loader = DataLoader(train_set,batch_size=64,sampler=sampler)
eval_loader = DataLoader(eval_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

subset_train_loader = DataLoader(subset_train, sampler=sampler_subset)
subset_eval_loader = DataLoader(subset_eval)
subset_test_loader = DataLoader(subset_test)

patches, labels = next(iter(train_loader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = ResNetPatchClassifier(
    n_object_classes=2,
    backbone="resnet18",
    pretrained=True,
    binary_mode="softmax"   # <---
).to(device)
learning_rate = 0.00001
opt=optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

loss_fn = FocalLoss(alpha=0.65, gamma=3.0)

epochs = 2
save_dir = "checkpoints_resnet"
os.makedirs(save_dir, exist_ok=True)

# Training loop
best_val_acc = 0.0
CONF_THRESHOLD = 0.85

for epoch in tqdm(range(epochs)):
    # Initialize epoch metrics counters for TRAIN
    train_tp, train_fp, train_fn, train_total = 0, 0, 0, 0
    train_avg_loss = 0.0

    # --------------------
    # TRAIN
    # --------------------
    model.train()
    print(f'* Epoch {epoch+1}/{epochs}')

    for patch, y_true in train_loader:
        patch = patch.to(device)
        y_true = y_true.to(device, dtype=torch.long)

        opt.zero_grad()
        y_pred_logits = model(patch)
        loss = loss_fn(y_pred_logits, y_true)

        loss.backward()
        opt.step()

        train_avg_loss += loss.item() / len(train_loader)
        
        # --- TRAINING PREDICTION (You can keep argmax here or change it) ---
        # It is usually okay to keep argmax in training to see how the model learns naturally.
        y_pred = y_pred_logits.argmax(dim=1)
        train_tp += ((y_pred == 1) & (y_true == 1)).sum().item()
        train_fp += ((y_pred == 1) & (y_true == 0)).sum().item()
        train_fn += ((y_pred == 0) & (y_true == 1)).sum().item()
        train_total += y_true.size(0)
        # ---------------------------

    train_acc = (train_total - train_fn - train_fp) / train_total # TN is not needed for Recall/Precision but for Acc
    
    # Add a small epsilon to prevent ZeroDivisionError
    epsilon = 1e-7
    
    train_precision = train_tp / (train_tp + train_fp + epsilon)
    train_recall = train_tp / (train_tp + train_fn + epsilon)
    
    print(f'  train_loss={train_avg_loss:.3f}, train_acc={train_acc:.3f}')
    print(f'  [TRAIN] Precision (Pothole)={train_precision:.3f}, Recall (Pothole)={train_recall:.3f}')


    model.eval()
    val_tp, val_fp, val_fn, val_total = 0, 0, 0, 0
    val_avg_loss = 0.0

    with torch.no_grad():
        for patch, y_true in eval_loader:
            patch = patch.to(device)
            y_true = y_true.to(device, dtype=torch.long)

            y_pred_logits = model(patch)
            loss = loss_fn(y_pred_logits, y_true)

            val_avg_loss += loss.item() / len(eval_loader)

            # ### CHANGE 1: Calculate Softmax Probabilities ###
            probs = torch.softmax(y_pred_logits, dim=1)
            
            # ### CHANGE 2: Get Probability of Pothole (Class 1) ###
            pothole_probs = probs[:, 1]
            
            # ### CHANGE 3: Apply Strict Threshold instead of argmax ###
            # If prob > 0.90, predict 1. Otherwise predict 0.
            y_pred = (pothole_probs > CONF_THRESHOLD).long()
            
            # --- Metric Accumulation ---
            val_tp += ((y_pred == 1) & (y_true == 1)).sum().item()
            val_fp += ((y_pred == 1) & (y_true == 0)).sum().item()
            val_fn += ((y_pred == 0) & (y_true == 1)).sum().item()
            val_total += y_true.size(0)
            # ---------------------------

    # --- Calculate and Print EVAL Metrics ---
    val_acc = (val_total - val_fn - val_fp) / val_total
    
    val_precision = val_tp / (val_tp + val_fp + epsilon)
    val_recall = val_tp / (val_tp + val_fn + epsilon)

    print(f'  val_loss={val_avg_loss:.3f}, val_acc={val_acc:.3f}')
    print(f'  [EVAL] Precision (Pothole)={val_precision:.3f}, Recall (Pothole)={val_recall:.3f}')

    # Option 2 (also): keep only the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print(f" New best model saved with val_acc={best_val_acc:.3f}")