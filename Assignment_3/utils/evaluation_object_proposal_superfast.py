############################################################
#   ULTRA-FAST PROPOSAL EVALUATION PIPELINE (GPU VERSION)
############################################################

import os
import time
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from create_boxes_edge import box_proposal_edge
from create_boxes_selective_search import box_proposal_ss
from visualize import read_content   # <-- your GT XML parser


############################################################
# 1. ULTRA-FAST IoU using PyTorch (GPU if available)
############################################################

def iou_matrix_torch(gt_boxes, prop_boxes):
    """
    Compute IoU between all GT and proposal boxes:
        gt_boxes:  (T,4)
        prop_boxes: (P,4)
    Returns:
        IoU matrix (T,P), on GPU if available.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt = torch.tensor(gt_boxes, dtype=torch.float32, device=device)      # (T,4)
    pb = torch.tensor(prop_boxes, dtype=torch.float32, device=device)    # (P,4)

    # Broadcast:
    gt = gt[:, None, :]   # (T,1,4)
    pb = pb[None, :, :]   # (1,P,4)

    # Intersection corners
    inter_xmin = torch.max(gt[..., 0], pb[..., 0])
    inter_ymin = torch.max(gt[..., 1], pb[..., 1])
    inter_xmax = torch.min(gt[..., 2], pb[..., 2])
    inter_ymax = torch.min(gt[..., 3], pb[..., 3])

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas
    gt_area = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    pb_area = (pb[..., 2] - pb[..., 0]) * (pb[..., 3] - pb[..., 1])

    union = gt_area + pb_area - inter_area + 1e-6

    return inter_area / union   # (T,P)


def compute_detection_rate_torch(gt_boxes, prop_boxes, k):

    # Force correct shapes
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    prop_boxes = np.asarray(prop_boxes, dtype=np.float32).reshape(-1, 4)

    if gt_boxes.shape[0] == 0 or prop_boxes.shape[0] == 0:
        return 0.0  # no GT or no proposals

    ious = iou_matrix_torch(gt_boxes, prop_boxes)
    detected = (ious > k).any(dim=1).float().mean().item()

    return detected


############################################################
# 4. Evaluation Function
############################################################

def eval_method(path, method, k, max_num):
    percentages = []
    xml_files = sorted(os.listdir(os.path.join(path, "annotations")))

    for xml_file in tqdm(xml_files):
        filename, true_boxes = read_content(os.path.join(path, "annotations", xml_file))
        image_path = os.path.join(path, "images", filename)

        proposals = method(image_path, N=max_num)
        perc = compute_detection_rate_torch(true_boxes, proposals, k)
        percentages.append(perc)

    return np.mean(percentages)

############################################################
# 6. RUN EVALUATION
############################################################

if __name__ == "__main__":

    if torch.cuda.is_available():
        print(f"🔥 CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available — using CPU.")

    path = "/dtu/datasets1/02516/potholes"
    k = 0.6
    max_num = 500

    # Choose method:
    method = box_proposal_ss
    # method = box_proposal_edge

    score = eval_method(path, method, k, max_num)
    print("\n====================================")
    print(f"Method: {method} | Number of proposal: {max_num} | k: {k}")
    print("Average detection rate:", score)
    print("====================================\n")
