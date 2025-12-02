import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms as torchvision_nms 
import torchvision.transforms as transforms



def iou(box1, box2):
    # Calculates Intersection over Union (IoU) of two bounding boxes.
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
    union = area1 + area2 - inter_area

    return inter_area / union


def non_max_suppression_manual(boxes, scores, iou_threshold):
    # Implements Non-Maximum Suppression (NMS) using NumPy.
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Sort boxes by confidence score (descending)
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate IoU between the current box (i) and the remaining boxes
        ious = []
        for j in order[1:]:
            ious.append(iou(boxes[i], boxes[j])) 

        ious = np.array(ious)

        # Filter out boxes with IoU > threshold (suppress them)
        inds_to_keep = np.where(ious <= iou_threshold)[0]
        order = order[inds_to_keep + 1] 
        
    return np.array(keep, dtype=np.int32)


# AP Calculation (Metric) 
def calculate_average_precision(all_detections, all_ground_truths, iou_threshold=0.5):
    
    # Aggregate predictions and initialize GT tracking
    all_pred_boxes = []
    all_pred_scores = []
    total_gt = 0
    gt_matched_flags = {} 

    for det, gt in zip(all_detections, all_ground_truths):
        total_gt += len(gt['boxes'])
        if det['boxes'].numel() > 0:
            all_pred_boxes.append(det['boxes'].numpy())
            all_pred_scores.append(det['scores'].numpy())
            gt_matched_flags[det['filename']] = [False] * len(gt['boxes'])

    if total_gt == 0:
        return 1.0, 0.0, 0.0, np.array([1.0]), np.array([0.0])
    if not all_pred_scores:
        return 0.0, 0.0, 0.0, np.array([0.0]), np.array([1.0])

    # Sort all predictions by score globally
    all_pred_boxes = np.concatenate(all_pred_boxes)
    all_pred_scores = np.concatenate(all_pred_scores)
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    sorted_boxes = all_pred_boxes[sorted_indices]
    
    sorted_filenames = [det['filename'] for det in all_detections for _ in det['boxes']]
    sorted_filenames = np.array(sorted_filenames)[sorted_indices] 
    
    # Matching and TP/FP/FN accumulation
    TP = np.zeros(len(sorted_boxes))
    FP = np.zeros(len(sorted_boxes))
    
    for i, pred_box in enumerate(sorted_boxes):
        filename = sorted_filenames[i]
        gt_index = [j for j, gt in enumerate(all_ground_truths) if gt['filename'] == filename][0]
        gt_boxes = all_ground_truths[gt_index]['boxes'].numpy()
        
        if len(gt_boxes) > 0:
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                current_iou = iou(pred_box, gt_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and not gt_matched_flags[filename][best_gt_idx]:
                TP[i] = 1.0 
                gt_matched_flags[filename][best_gt_idx] = True 
            else:
                FP[i] = 1.0 
        else:
            FP[i] = 1.0 

    # Precision-Recall Curve Calculation
    TP_cum = np.cumsum(TP)
    FP_cum = np.cumsum(FP)
    
    Precision = TP_cum / (TP_cum + FP_cum)
    Recall = TP_cum / total_gt
    
    # Statistics
    max_recall = Recall.max() if Recall.size > 0 else 0.0
    recall_at_50_idx = np.where(Recall >= 0.5)[0]
    P50 = Precision[recall_at_50_idx[0]] if recall_at_50_idx.size > 0 else 0.0
    
    # AP Calculation (11-point interpolation)
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1): 
        max_p = 0.0
        if Recall.size > 0:
             max_p = np.max(Precision[Recall >= t]) if np.sum(Recall >= t) > 0 else 0.0
        ap += max_p / 11.0
        
    return ap, max_recall, P50, Precision, Recall # Return P/R vectors for plotting


# Evaluation Functions 

# Global constants needed for both functions 
# PROPOSALS_PER_IMAGE, SCORE_THRESHOLD, NMS_IOU_THRESHOLD, PATCH_SIZE, transform_test, device

def evaluate_single_image_manual(model, filename, PROPOSALS_PER_IMAGE, SCORE_THRESHOLD, NMS_IOU_THRESHOLD, PATCH_SIZE, transform_test, device, box_proposal_ss, DATASET_PATH):
    # Applies Selective Search, CNN classification, and MANUAL NMS (NumPy/CPU).
    img_path = os.path.join(DATASET_PATH, "images", filename)
    proposal_boxes = box_proposal_ss(img_path, N=PROPOSALS_PER_IMAGE) 
    img = Image.open(img_path).convert("RGB")
    all_patches = []
    
    # Cropping and Transformation
    for (x1, y1, x2, y2) in proposal_boxes:
        crop = img.crop((x1, y1, x2, y2))
        crop = transforms.Resize(PATCH_SIZE)(crop)
        patch_tensor = transform_test(crop)
        all_patches.append(patch_tensor)
    
    if not all_patches:
        return torch.empty(0, 4), torch.empty(0) 
        
    patches_batch = torch.stack(all_patches).to(device)
    
    with torch.no_grad():
        logits = model(patches_batch) 
        probs = torch.softmax(logits, dim=1) 
        pothole_probs = probs[:, 1]
        
    # Filtering
    high_conf_indices = (pothole_probs >= SCORE_THRESHOLD).nonzero(as_tuple=True)[0]
    
    if high_conf_indices.numel() == 0:
        return torch.empty(0, 4), torch.empty(0)

    # Prepare for MANUAL NMS 
    filtered_boxes_np = proposal_boxes[high_conf_indices.cpu()].reshape(-1, 4)
    filtered_scores_np = pothole_probs[high_conf_indices].cpu().numpy() 

    # Apply MANUAL NMS
    keep_indices_np = non_max_suppression_manual(
        filtered_boxes_np, 
        filtered_scores_np, 
        NMS_IOU_THRESHOLD
    )
    
    final_boxes_nms = filtered_boxes_np[keep_indices_np]
    final_scores_nms = filtered_scores_np[keep_indices_np]
    
    return torch.tensor(final_boxes_nms, dtype=torch.float32), torch.tensor(final_scores_nms, dtype=torch.float32)


def evaluate_single_image_torch(model, filename, PROPOSALS_PER_IMAGE, SCORE_THRESHOLD, NMS_IOU_THRESHOLD, PATCH_SIZE, transform_test, device, box_proposal_ss, DATASET_PATH):
    # Applies Selective Search, CNN classification, and TORCHVISION NMS (CUDA/GPU).
    img_path = os.path.join(DATASET_PATH, "images", filename)
    proposal_boxes = box_proposal_ss(img_path, N=PROPOSALS_PER_IMAGE) 
    
    img = Image.open(img_path).convert("RGB")
    all_patches = []
    
    # Cropping and Transformation (identical)
    for (x1, y1, x2, y2) in proposal_boxes:
        crop = img.crop((x1, y1, x2, y2))
        crop = transforms.Resize(PATCH_SIZE)(crop)
        patch_tensor = transform_test(crop)
        all_patches.append(patch_tensor)
    
    if not all_patches:
        return torch.empty(0, 4), torch.empty(0) 
        
    patches_batch = torch.stack(all_patches).to(device)
    
    with torch.no_grad():
        logits = model(patches_batch) 
        probs = torch.softmax(logits, dim=1) 
        pothole_probs = probs[:, 1]
        
    # Filtering
    high_conf_indices = (pothole_probs >= SCORE_THRESHOLD).nonzero(as_tuple=True)[0]
    
    if high_conf_indices.numel() == 0:
        return torch.empty(0, 4), torch.empty(0)

    # Prepare for TORCHVISION NMS 
    filtered_boxes_np = proposal_boxes[high_conf_indices.cpu()].reshape(-1, 4)

    final_boxes = torch.tensor(filtered_boxes_np, dtype=torch.float32).to(device)   
    final_scores = pothole_probs[high_conf_indices].to(device) 

    # Apply TORCHVISION NMS
    keep_indices = torchvision_nms(final_boxes, final_scores, NMS_IOU_THRESHOLD)
    
    final_boxes_nms = final_boxes[keep_indices]
    final_scores_nms = final_scores[keep_indices]

    return final_boxes_nms.cpu(), final_scores_nms.cpu()


# 4. Visualization Functions

def visualize_single(image_path, pred_boxes, gt_boxes, save_dir=None):
    # Visualizes image with predicted boxes (red) and Ground Truth boxes (blue).
    
    # Load image and create plot
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw PREDICTIONS (Red)
    for box in pred_boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='red', facecolor='none', label='Prediction')
        ax.add_patch(rect)

    # Draw GROUND TRUTH (Blue)
    for box in gt_boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='blue', facecolor='none', label='Ground Truth')
        ax.add_patch(rect)

    # Labels and Save
    plt.title(os.path.basename(image_path))
    plt.axis("off")
    
    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.savefig(os.path.join("images", os.path.basename(image_path)))
    
    plt.close(fig)