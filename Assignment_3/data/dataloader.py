from glob import glob
import os
import pandas as pd # Not strictly used in this class, but kept from original imports
from PIL import Image
import torch
from torchvision import transforms as T # Not strictly used in this class, but kept from original imports
from torch.utils.data import Sampler # Not strictly used in this class, but kept from original imports
import numpy as np # Not strictly used in this class, but kept from original imports
from utils.visualize import read_content
from utils.create_boxes_edge import box_proposal
from torch.utils.data import Dataset

DATA_PATH = '/zhome/fe/3/214307/pythonHPC/project4DL/data/patholes'
DATA_PATH_ORIGINAL_IMAGE = '/zhome/fe/3/214307/pythonHPC/project4DL/data/patholes' #the other one
DATA_ROOT = Path("/dtu/datasets1/02516/potholes")
IMG_DIR = DATA_ROOT / "images"

def getSplit(split):
    images = sorted(IMG_DIR.glob("*.png"))

    valid_filenames = []
    missing_xml = 0

    for img_path in images:
        xml_name = img_path.name.replace(".png", ".xml")
        xml_path = ANN_DIR / xml_name
        if xml_path.exists():
            valid_filenames.append(img_path.name)
        else:
            missing_xml += 1

    # shuffle the dataset for randomness
    random.seed(42)
    random.shuffle(valid_filenames)

    n_total = len(valid_filenames)

    # Compute split sizes: 70% train, 15% val, 15% test
    # or 70, 20 and 10
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val  # remaining samples

    # Create the splits
    train_files = valid_filenames[:n_train]
    val_files   = valid_filenames[n_train : n_train + n_val]
    test_files  = valid_filenames[n_train + n_val :]

    train_names = [os.path.basename(p) for p in train_paths]
    val_names   = [os.path.basename(p) for p in val_paths]
    test_names  = [os.path.basename(p) for p in test_paths]

    if split == "train": 
        return train_names
    elif split == "test": 
        return test_names 
    else : 
        return val_names 

DATA_PATH = '/zhome/fe/3/214307/pythonHPC/project4DL/data/patholes'


class PHC2(Dataset):
    def __init__(self, split, transform=None, iou_threshold=0.5, proposals_per_image=50):
        self.split = split
        self.transform = transform

        # list of image file names, e.g. ["img_001.jpg", ...]
        self.image_names = getSplit(split)

        self.root = DATA_PATH

        # build a list of (image_path, proposal_box, label)
        self.samples = []
        for name in self.image_names:
            img_path = os.path.join(self.root, name)

            # 1) read ground-truth boxes from XML
            xml_path = os.path.splitext(img_path)[0] + ".xml"
            gt_boxes = read_content(xml_path)   # e.g. list of [xmin, ymin, xmax, ymax]

            # 2) get proposals from your algorithm
            proposals = box_proposal_edge(img_path, N=proposals_per_image)

            self.samples.append((img_path, proposals, gt_boxes))


    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, box_list, gt_boxes = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        crops = []
        for (xmin, ymin, xmax, ymax) in box_list:
            crop = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))  # PIL

            # apply your existing transforms first (e.g. ToTensor, Normalize, etc.)
            if self.transform is not None:
                crop = self.transform(crop)   # tensor [C, H, W]
            else:
                crop = transforms.ToTensor()(crop)

            C, H, W = crop.shape

            # Optionally: if the crop is larger than 32 in any dimension, resize down
            # (if you know all boxes are smaller than 32 you can skip this block)
            if H > 256 or W > 256:
                # keep aspect ratio or just force 32x32; simplest:
                crop = F.interpolate(
                    crop.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
                ).squeeze(0)
                C, H, W = crop.shape

            # now pad to 32x32 with zeros (black)
            pad_h = 256 - H
            pad_w = 256 - W

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
            crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

            # now crop is [C, 32, 32]
            crops.append(crop)

        crops = torch.stack(crops, dim=0)  # [num_proposals, C, 32, 32]

        boxes_tensor = torch.tensor(box_list, dtype=torch.float32)   # [num_proposals, 4]
        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)  # [num_gt, 4]

        return crops, boxes_tensor, gt_boxes_tensor

