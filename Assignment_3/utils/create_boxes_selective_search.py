import cv2
import numpy as np
from PIL import Image
from visualize import visualize_single
import time

# Selective Search function
def box_proposal_ss(image_path, mode="fast", max_proposals=50):

    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil).astype(np.float32) / 255.0

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    start = time.time()

    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()  # list of (x, y, w, h)

    if max_proposals is not None and len(rects) > max_proposals:
        rects = rects[:max_proposals]

    end = time.time()
    elapsed = end - start

    print(f"EdgeBoxes processing time: {elapsed:.4f} seconds")

    return rects

if __name__=="__main__":

    image_path = "/dtu/datasets1/02516/potholes/images/potholes1.png"
    boxes = box_proposal_ss(image_path)

    visualize_single(image_path, boxes, save_dir="box_proposa_ss.png")

    print(boxes)


