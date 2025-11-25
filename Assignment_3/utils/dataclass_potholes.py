import json
import torch
from dataclasses import dataclass
from typing import List
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


@dataclass
class Potholes:
    filename: str                  # image filename
    width: int                     # image width
    height: int                    # image height
    proposals: torch.Tensor        # shape (N, 4)
    labels: torch.Tensor           # shape (N,)


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

class PotholeDataset(Dataset):
    def __init__(self, proposal_dir, image_dir, transform=None):
        self.proposal_dir = proposal_dir
        self.image_dir = image_dir
        self.transform = transform

        # Load all dataclass objects
        self.samples = []
        for f in sorted(os.listdir(proposal_dir)):
            if f.endswith(".json"):
                self.samples.append(load_pothole_json(os.path.join(proposal_dir, f)))

        # Build global proposal-level index
        self.index = []   # list of (sample_idx, proposal_idx)
        for si, sample in enumerate(self.samples):
            for pi in range(sample.proposals.shape[0]):
                self.index.append((si, pi))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, pi = self.index[idx]
        sample = self.samples[si]

        # Load raw image
        img_path = os.path.join(self.image_dir, sample.filename)
        img = Image.open(img_path).convert("RGB")

        # Extract proposal box
        x1, y1, x2, y2 = sample.proposals[pi].tolist()
        patch = img.crop((x1, y1, x2, y2))

        # Apply transform
        if self.transform:
            patch = self.transform(patch)

        label = sample.labels[pi]

        return patch, label

if __name__=="__main__":

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = PotholeDataset(
        proposal_dir="dataset_proposal",
        image_dir="/dtu/datasets1/02516/potholes/images",
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_idx, (patches, labels) in enumerate(loader):
        print("Batch index:", batch_idx)
        print("Patches shape:", patches.shape)
        print("Labels:", labels)
        print("Unique labels in batch:", labels.unique())
        break  # remove to see more batches

    plt.figure(figsize=(10,4))
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(F.to_pil_image(patches[i]))
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("Examples.png")


