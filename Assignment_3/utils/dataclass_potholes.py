import json
import torch
from dataclasses import dataclass
from typing import List
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


@dataclass
class Potholes:
    filename: str
    width: int
    height: int
    proposals: torch.Tensor
    labels: torch.Tensor


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
    def __init__(self, proposal_dir, image_dir, transform=None,
                 split=(70, 20, 10), mode="train"):

        assert sum(split) == 100, "Split must sum to 100"
        assert mode in ["train", "eval", "test"], "mode must be train/eval/test"

        self.proposal_dir = proposal_dir
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

        # --------------------------------------------------
        # 1. Load all JSON filenames (sorted = deterministic split)
        # --------------------------------------------------
        all_json = sorted([f for f in os.listdir(proposal_dir) if f.endswith(".json")])
        total = len(all_json)

        # Compute split ranges
        train_end = int(split[0] / 100 * total)
        eval_end = train_end + int(split[1] / 100 * total)

        if mode == "train":
            json_files = all_json[:train_end]
        elif mode == "eval":
            json_files = all_json[train_end:eval_end]
        else:  # test
            json_files = all_json[eval_end:]


        self.samples = [
            load_pothole_json(os.path.join(proposal_dir, f))
            for f in json_files
        ]

        self.index = []
        for si, sample in enumerate(self.samples):
            for pi in range(sample.proposals.shape[0]):
                self.index.append((si, pi))

    def __len__(self):
        return len(self.index)

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


# ===========================================================
# Example usage
# ===========================================================
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # Example: load TRAIN SET
    train_set = PotholeDataset(
        proposal_dir="dataset_proposal",
        image_dir="/dtu/datasets1/02516/potholes/images",
        transform=transform,
        split=(70, 20, 10),
        mode="train"
    )

    loader = DataLoader(train_set, batch_size=32, shuffle=True)

    for batch_idx, (patches, labels) in enumerate(loader):
        print("Batch index:", batch_idx)
        print("Patches shape:", patches.shape)
        print("Labels:", labels)
        print("Unique labels in batch:", labels.unique())
        break

    plt.figure(figsize=(10,4))
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.imshow(F.to_pil_image(patches[i]))
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("Examples.png")

    print("Total samples:", len(train_set))
    print("Batch size:", loader.batch_size)
    print("Expected batches:", len(loader))
