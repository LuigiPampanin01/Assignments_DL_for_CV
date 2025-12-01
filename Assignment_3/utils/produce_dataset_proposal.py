import os
from create_boxes_selective_search import box_proposal_ss
from tqdm import tqdm
import numpy as np
from evaluation_object_proposal_superfast import iou_matrix_torch
from PIL import Image
from visualize import read_content
import json


def assign_labels_with_iou_torch(proposals, gt_boxes, k):
    """
    Assign a binary label (1 = positive, 0 = background) to each proposal box
    based on its IoU with the ground-truth bounding boxes.

    -------------------------------------------
    Inputs:
        proposals : list or array of shape (P, 4)
            Proposed region boxes from Selective Search or EdgeBoxes.
            Format: [xmin, ymin, xmax, ymax]

        gt_boxes : list or array of shape (T, 4)
            Ground-truth bounding boxes (true potholes).

        k : float
            IoU threshold. A proposal is labeled positive if its IoU with
            ANY ground-truth box is >= k.

    -------------------------------------------
    Output:
        labels : list of length P
            labels[i] = 1  → proposal i overlaps a pothole with IoU ≥ k
            labels[i] = 0  → proposal i is background
    -------------------------------------------

    Notes:
    - Multiple proposals can be positive for the same object. This is GOOD.
    - A proposal that overlaps multiple GT boxes still gets label = 1.
    - This matches the R-CNN training pipeline exactly.
    """

    # If there are no proposals at all, return empty labels.
    if len(proposals) == 0:
        return []

    # Convert input lists into (P,4) and (T,4) float32 NumPy arrays.
    # Ensures consistent shape and avoids type errors.
    proposals = np.array(proposals, dtype=np.float32).reshape(-1, 4)
    gt_boxes  = np.array(gt_boxes, dtype=np.float32).reshape(-1, 4)

    # Compute full IoU matrix using GPU if available.
    # ious has shape (T, P):
    #   ious[t, p] = IoU between GT box t and proposal p.
    ious = iou_matrix_torch(gt_boxes, proposals).cpu().numpy()

    # For each proposal p, we take the maximum IoU it has with ANY GT box.
    # Example: if proposal p overlaps multiple GT objects,
    # best_iou[p] will be the highest IoU among them.
    best_iou = ious.max(axis=0)   # shape (P,)

    # A proposal is considered positive if it overlaps ANY GT box
    # with IoU >= k. Otherwise, label = 0 (background).
    labels = (best_iou >= k).astype(int).tolist()

    return labels


def save_json(path, filename, img_w, img_h, proposals, labels):
    data = {
        "filename": filename,
        "width": img_w,
        "height": img_h,
        "proposals": proposals,
        "labels": labels
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


############################################################
# MASTER FUNCTION YOU ASKED FOR
############################################################

def generate_and_save_proposals(dataset_path, save_dir, method, k=0.5, max_num=1000):
    """
    dataset_path: root folder containing /images and /annotations
    save_dir: where to store .json proposal files
    method: the proposal function (SS or EdgeBoxes)
    k: IoU threshold
    max_num: number of proposals per image
    """

    os.makedirs(save_dir, exist_ok=True)

    annotation_dir = os.path.join(dataset_path, "annotations")
    image_dir = os.path.join(dataset_path, "images")

    xml_files = sorted(os.listdir(annotation_dir))

    print(f"\n🔥 Generating proposal JSONs into: {save_dir}")
    print(f"➡ Method: {method.__name__}, proposals: {max_num}, k={k}\n")

    for xml_file in tqdm(xml_files):
        # --- Load GT ---
        filename, gt_boxes = read_content(os.path.join(annotation_dir, xml_file))
        img_path = os.path.join(image_dir, filename)

        # --- Load image for size ---
        img = Image.open(img_path)
        img_w, img_h = img.size

        # --- Generate proposals ---
        proposals = method(img_path, N=max_num)   # (P,4)

        # Convert to list-of-lists (JSON-safe)
        proposals = proposals.tolist()

        # --- Assign labels ---
        labels = assign_labels_with_iou_torch(proposals, gt_boxes, k)

        # --- Save JSON ---
        json_name = filename.replace(".png", ".json").replace(".jpg", ".json")
        json_path = os.path.join(save_dir, json_name)

        save_json(json_path, filename, img_w, img_h, proposals, labels)

    print("\nDONE! All proposal JSONs saved.\n")

if __name__=="__main__":

    dataset_path = "/dtu/datasets1/02516/potholes"
    save_dir = "dataset_proposal2"

    generate_and_save_proposals(
    dataset_path=dataset_path,
    save_dir=save_dir,
    method=box_proposal_ss,
    k=0.5,
    max_num=100
)