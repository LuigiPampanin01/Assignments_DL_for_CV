import json
import torch
from dataclasses import dataclass
from typing import List
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


# ============================================================
# Dataclass for storing proposals of a single image
# ============================================================
@dataclass
class Potholes:
    filename: str
    width: int
    height: int
    proposals: torch.Tensor   # shape (N, 4)
    labels: torch.Tensor      # shape (N,)


# ============================================================
# Load JSON → into dataclass
# ============================================================
def load_pothole_json(json_path: str) -> Potholes:
    with open(json_path, "r") as f:
        d = json.load(f)
    return Potholes(
        filename=d["filename"],
        width=d["width"],
        height=d["height"],
        proposals=torch.tensor(d["proposals"], dtype=torch.float32),
        labels=torch.tensor(d["labels"], dtype=torch.long)
    )


# ============================================================
# Main Dataset Class (supports splitting)
# ============================================================
class PotholeDataset(Dataset):
    def __init__(self, proposal_dir, image_dir, transform=None,
                 split=(70, 20, 10), mode="train"):

        assert sum(split) == 100, "Split must sum to 100"
        assert mode in ["train", "eval", "test"], "mode must be 'train', 'eval', or 'test'"

        self.proposal_dir = proposal_dir
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

        # ------------------------------
        # 1. Deterministic image-level split
        # ------------------------------
        all_json = sorted([f for f in os.listdir(proposal_dir) if f.endswith(".json")])
        total = len(all_json)

        train_end = int(split[0] / 100 * total)
        eval_end = train_end + int(split[1] / 100 * total)

        if mode == "train":
            selected = all_json[:train_end]
        elif mode == "eval":
            selected = all_json[train_end:eval_end]
        else:
            selected = all_json[eval_end:]

        # ------------------------------
        # 2. Load selected images as dataclasses
        # ------------------------------
        self.samples = [
            load_pothole_json(os.path.join(proposal_dir, f))
            for f in selected
        ]

        # ------------------------------
        # 3. Build proposal-level index
        # ------------------------------
        self.index = []
        for si, sample in enumerate(self.samples):
            for pi in range(sample.proposals.shape[0]):
                self.index.append((si, pi))


    # Length of dataset = total number of proposals
    def __len__(self):
        return len(self.index)


    # Return one (patch, label)
    def __getitem__(self, idx):
        si, pi = self.index[idx]
        sample = self.samples[si]

        img_path = os.path.join(self.image_dir, sample.filename)
        img = Image.open(img_path).convert("RGB")

        x1, y1, x2, y2 = sample.proposals[pi].tolist()
        patch = img.crop((x1, y1, x2, y2))

        if self.transform:
            patch = self.transform(patch)

        label = sample.labels[pi]
        return patch, label



# ============================================================
# Helper: create a WeightedRandomSampler to balance classes
# ============================================================
def create_balanced_sampler(dataset):
    all_labels = []

    # Collect labels in order of dataset.index
    for si, pi in dataset.index:
        all_labels.append(dataset.samples[si].labels[pi].item())

    labels_tensor = torch.tensor(all_labels)

    pos_count = (labels_tensor == 1).sum().item()
    neg_count = (labels_tensor == 0).sum().item()

    # sampling weights
    weight_pos = neg_count / pos_count           # > 1
    weight_neg = 1.0

    weights = torch.where(
        labels_tensor == 1,
        torch.tensor(weight_pos, dtype=torch.float),
        torch.tensor(weight_neg, dtype=torch.float)
    )

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)



# ============================================================
# Example Usage
# ============================================================
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # ----------------------------------------
    # Load TRAIN SET
    # ----------------------------------------
    train_set = PotholeDataset(
        proposal_dir="dataset_proposal",
        image_dir="/dtu/datasets1/02516/potholes/images",
        transform=transform,
        split=(70, 20, 10),
        mode="train"
    )

    sampler = create_balanced_sampler(train_set)

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        sampler=sampler
    )

    # ----------------------------------------
    # Inspect one batch
    # ----------------------------------------
    patches, labels = next(iter(train_loader))

    print("Batch shape:", patches.shape)
    print("Labels:", labels[:32])
    print("Unique labels:", labels.unique())

    # ----------------------------------------
    # Visualization
    # ----------------------------------------
    plt.figure(figsize=(20,8))
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.imshow(F.to_pil_image(patches[i]))
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("Examples.png")

    print("Total samples in TRAIN:", len(train_set))
