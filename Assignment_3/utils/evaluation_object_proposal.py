from visualize import read_content
from create_boxes_edge import box_proposal_edge
from create_boxes_selective_search import box_proposal_ss
import os
import numpy as np
from tqdm import tqdm

def eval_method(path, method, k, max_num):

    percentages = []

    xlm_files = os.listdir(os.path.join(path, "annotations"))

    for xlm in tqdm(xlm_files):

        filename, list_with_all_boxes = read_content(os.path.join(path, "annotations", xlm))

        image_path = os.path.join(path, "images", filename)

        boxes_proposal = method(image_path, N=max_num)

        perc = compute_detection_rate(list_with_all_boxes, boxes_proposal, k)

        percentages.append(perc)

    mean_perc = np.mean(np.array(percentages))

    return mean_perc

def compute_detection_rate(true_boxes, proposal_boxes, k):

    count = 0

    for true_box in true_boxes:
        for proposal_box in proposal_boxes:
            IoU = iou(true_box, proposal_box)
            if IoU > k:
                count = count + 1
                break

    perc = count / len(true_boxes)

    return perc

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection box
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # No overlap
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area1 + area2 - inter_area

    return inter_area / union


if __name__=="__main__":

    path = "/dtu/datasets1/02516/potholes"
    k = 0.8
    max_num = 200
    method = box_proposal_edge   # <-- FIXED (no parentheses)

    score = eval_method(path, method, k, max_num)
    print("Average detection rate:", score)
