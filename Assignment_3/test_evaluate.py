from model.model import ResNetPatchClassifier
import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from utils_evaluation import (
    calculate_average_precision, 
    evaluate_single_image_manual, 
    evaluate_single_image_torch, 
    visualize_single
)

from utils.create_boxes_selective_search import box_proposal_ss
from utils.visualize import read_content 

# 1. GLOBAL CONFIGURATION PARAMETERS 
DATASET_PATH = "/dtu/datasets1/02516/potholes"
CHECKPOINT_PATH = "checkpoints_resnet/best_model.pth"

# TWEAKABLE PARAMETERS FOR EXPERIMENTS 

PROPOSALS_PER_IMAGE = 1000           # Number of Proposals determines maximum possible Recall (Coverage)
SCORE_THRESHOLD = 0.5              # Confidence Threshold determines Precision/Recall trade-off (Filtering)
NMS_IOU_THRESHOLD = 0.3            # NMS IoU Threshold determines how aggressively overlapping boxes are suppressed
USE_MANUAL_NMS = False               # NMS Implementation choice (True for Manual/NumPy, False for PyTorch/CUDA)

# MODEL AND DATA SETUP 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = (224, 224) 

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_trained_model():
    model = ResNetPatchClassifier(
        n_object_classes=2,
        backbone="resnet18",
        pretrained=False,
        binary_mode="softmax"
    ).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint) 
    model.eval()
    print(f"Model loaded successfully from {CHECKPOINT_PATH}")
    return model

def get_test_image_names(dataset_path):
    # Replicates the 70/20/10 split logic to get the test set filenames
    from utils.dataclass_potholes import PotholeDataset
    
    test_file_names = PotholeDataset(
        proposal_dir=os.path.join(os.getcwd(), "utils/dataset_proposal2"), 
        image_dir=os.path.join(DATASET_PATH, "images"),
        mode="test"
    ).samples 
    
    return [s.filename for s in test_file_names]


def main_evaluation():
    # SETUP AND CONFIGURATION PRINT
    output_dir = "detection_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")
    
    model = load_trained_model()
    test_image_names = get_test_image_names(DATASET_PATH) 

    if USE_MANUAL_NMS:
        evaluation_func = evaluate_single_image_manual
        nms_mode = "Manual (NumPy/CPU)"
    else:
        evaluation_func = evaluate_single_image_torch
        nms_mode = "PyTorch (CUDA/GPU)"

    print("\n========================================")
    print("EVALUATION CONFIGURATION")
    print(f"NMS Mode: {nms_mode}")
    print(f"Num Proposals per Image: {PROPOSALS_PER_IMAGE}")
    print(f"Confidence Threshold (Score): {SCORE_THRESHOLD:.2f}")
    print(f"NMS IoU Threshold: {NMS_IOU_THRESHOLD:.2f}")
    print("========================================\n")
    
    print(f"Starting evaluation on {len(test_image_names)} test images...")

    all_detections = []
    all_ground_truths = []
    
    # EVALUATION LOOP
    for filename in tqdm(test_image_names):
        
        # Get predictions using the selected NMS function
        pred_boxes, pred_scores = evaluation_func(
            model, filename, PROPOSALS_PER_IMAGE, SCORE_THRESHOLD, NMS_IOU_THRESHOLD, 
            PATCH_SIZE, transform_test, device, box_proposal_ss, DATASET_PATH
        )
        
        # Get Ground Truth
        _, gt_boxes = read_content(os.path.join(DATASET_PATH, "annotations", filename.replace(".png", ".xml")))
        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

        # Visualization (First 10 images saved to check performance)
        if test_image_names.index(filename) < 10: 
            visualize_single(
                os.path.join(DATASET_PATH, "images", filename), 
                pred_boxes.tolist(), 
                gt_boxes, 
                save_dir=os.path.join(output_dir, f"result_{filename}")
            )

        # Collect Data for AP
        all_detections.append({'filename': filename, 'boxes': pred_boxes, 'scores': pred_scores})
        all_ground_truths.append({'filename': filename, 'boxes': gt_boxes_tensor})

    # FINAL METRICS AND OUTPUT 
    
    ap_score, max_recall, P50, Precision, Recall = calculate_average_precision(all_detections, all_ground_truths, iou_threshold=0.5) 
    
    print("\n--- FINAL RESULT ---")
    print(f"Predictions collected. Total images: {len(all_detections)}")
    print(f"Average Precision (AP) @ IoU 0.50: {ap_score:.3f}")
    print(f"Maximum Recall (Max Coverage): {max_recall:.3f}")
    print(f"Precision at 50% Recall (P50): {P50:.3f}")

    # SAVE TOP SAMPLES VISUALIZATION (we can use them for the report)
    
    all_files_and_scores = []

    # Aggregate all scores and track their origin
    for det in all_detections:
        for score in det['scores']:
            all_files_and_scores.append({
                'filename': det['filename'], 
                'score': score.item()
            })

    # Sort all predicted boxes by score globally
    all_files_and_scores.sort(key=lambda x: x['score'], reverse=True)

    # Select the top N unique images based on the highest scores
    TOP_N_IMAGES = 5
    
    top_files_to_save = []
    seen_files = set()

    for item in all_files_and_scores:
        if item['filename'] not in seen_files:
            top_files_to_save.append(item['filename'])
            seen_files.add(item['filename'])
            if len(top_files_to_save) >= TOP_N_IMAGES:
                break
    
    print(f"\nSelected the {TOP_N_IMAGES} images with the highest confidence predictions:")
    save_dir = os.path.join(output_dir, "TOP_SAMPLES")
    os.makedirs(save_dir, exist_ok=True) 

    for filename in top_files_to_save:
        pred_boxes, _ = next((d['boxes'], d['scores']) for d in all_detections if d['filename'] == filename)
        
        # Get Ground Truth (from XML)
        _, gt_boxes = read_content(os.path.join(DATASET_PATH, "annotations", filename.replace(".png", ".xml")))
        
        # Save visualization to TOP_SAMPLES folder
        visualize_single(
            os.path.join(DATASET_PATH, "images", filename), 
            pred_boxes.tolist(), 
            gt_boxes, 
            save_dir=os.path.join(save_dir, f"TOP_{filename}")
        )
        print(f"Saved {filename} to {save_dir}")

    # SAVE P/R CURVE 
    plt.figure(figsize=(8, 6))
    plt.plot(Recall, Precision, marker='.', label=f'AP @ 0.50 = {ap_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Object Detection)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "PR_Curve.png"))
    plt.close()

    print(f"\nPR Curve saved in {output_dir}/PR_Curve.png")

if __name__ == "__main__":
    main_evaluation()