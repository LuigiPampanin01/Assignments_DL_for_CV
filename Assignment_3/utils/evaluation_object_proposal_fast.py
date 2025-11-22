############################################################
# FAST OBJECT PROPOSAL EVALUATION PIPELINE (Numba-Optimized)
############################################################

import os
import time
import cv2
import numpy as np
from numba import njit
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualize import read_content   # <-- your own function to read GT XML

############################################################
# 1. Numba-Optimized IoU
############################################################

@njit(fastmath=True)
def iou_numba_single(box1, box2):
    """
    IoU between 2 boxes using Numba.
    Format: (xmin, ymin, xmax, ymax)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    return inter_area / (area1 + area2 - inter_area)


@njit(fastmath=True)
def iou_numba_batch(gt_box, proposals):
    """
    Compute IoU of 1 ground-truth box vs many proposals (Numba).
    """
    N = proposals.shape[0]
    result = np.zeros(N)
    for i in range(N):
        result[i] = iou_numba_single(gt_box, proposals[i])
    return result


def compute_detection_rate_fast(true_boxes, proposal_boxes, k):
    """
    Fast detection rate using Numba-accelerated IoU.
    """
    true_boxes = np.asarray(true_boxes, dtype=np.float32)
    proposal_boxes = np.asarray(proposal_boxes, dtype=np.float32)

    detected = 0
    for gt in true_boxes:
        ious = iou_numba_batch(gt, proposal_boxes)
        if np.any(ious > k):
            detected += 1

    return detected / len(true_boxes)

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

        perc = compute_detection_rate_fast(true_boxes, proposals, k)
        percentages.append(perc)

    return np.mean(percentages)


############################################################
# 5. Visualization
############################################################

def visualize_single(image_path, boxes, save_path=None):
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(img)

    for (xmin, ymin, xmax, ymax) in boxes:
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


############################################################
# 6. Run Evaluation
############################################################

if __name__ == "__main__":
    path = "/dtu/datasets1/02516/potholes"
    k = 0.8
    max_num = 200

    method = box_proposal_edge   # or box_proposal_ss

    score = eval_method(path, method, k, max_num)
    print("Average detection rate:", score)
