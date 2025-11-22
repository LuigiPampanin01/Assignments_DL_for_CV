import cv2
import numpy as np
from PIL import Image
import time
from visualize import visualize_single

def box_proposal_ss(image_path, mode="fast", N=100):

    # Load as uint8, because Selective Search requires it
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil, dtype=np.uint8)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    start = time.time()

    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()  # (x, y, w, h)

    # Limit to top N
    if N is not None:
        rects = rects[:N]

    # Convert to (xmin, ymin, xmax, ymax)
    converted = []
    for (x, y, w, h) in rects:
        converted.append((x, y, x + w, y + h))

    elapsed = time.time() - start
    # print(f"Selective Search processing time: {elapsed:.4f} seconds")

    return np.array(converted, dtype=np.float32)


if __name__ == "__main__":
    image_path = "/dtu/datasets1/02516/potholes/images/potholes1.png"
    boxes = box_proposal_ss(image_path)

    visualize_single(image_path, boxes, save_dir="box_proposal_ss.png")
