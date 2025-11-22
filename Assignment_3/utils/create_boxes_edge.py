import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from visualize import visualize_single

def box_proposal_edge(image_path, N=100):

    # Load PIL → numpy float32 → normalized RGB
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil).astype(np.float32) / 255.0


    # Initialize EdgeBoxes + model
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(N)
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model/model.yml.gz")

    start = time.time()

    # Run full EdgeBoxes pipeline
    edges = edge_detector.detectEdges(img)
    orimap = edge_detector.computeOrientation(edges)
    edges_nms = edge_detector.edgesNms(edges, orimap)
    boxes, scores = edge_boxes.getBoundingBoxes(edges_nms, orimap)

    converted = []
    for (x, y, w, h) in boxes:
        converted.append((x, y, x + w, y + h))

    end = time.time()
    elapsed = end - start

    # print(f"EdgeBoxes processing time: {elapsed:.4f} seconds")

    return np.array(converted, dtype=np.float32)

if __name__=="__main__":

    image_path = "/dtu/datasets1/02516/potholes/images/potholes1.png"
    boxes = box_proposal_edge(image_path)

    visualize_single(image_path, boxes, save_dir="box_proposal_edge.png")
