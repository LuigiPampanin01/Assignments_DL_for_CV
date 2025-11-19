import cv2
import numpy as np
from pathlib import Path
import json
import time

# Paths
DATA_ROOT = Path("/dtu/datasets1/02516/potholes")
IMG_DIR   = DATA_ROOT / "images"

SPLITS_PATH = Path.home() / "IDLCV_Project4" / "splits.json"
OUT_DIR     = Path.home() / "IDLCV_Project4" / "proposals_ss"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Selective Search function
def run_selective_search(img, mode="fast", max_proposals=2000):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()  # list of (x, y, w, h)

    if max_proposals is not None and len(rects) > max_proposals:
        rects = rects[:max_proposals]

    return np.array(rects, dtype=np.int32)



# Process a single image
def process_image(filename, resize_to=400):
    
    img_path = IMG_DIR / filename
    out_path = OUT_DIR / f"{Path(filename).stem}_ss.npz"

    if out_path.exists():
        out_path.unlink()  # delete old file
        print(f"[OVERWRITE] Removed old {out_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Could not load: {img_path}")
        return

    orig_h, orig_w = img.shape[:2]
    scale_x = scale_y = 1.0

    # Resize image for efficiency (if resize_to is given)
    if resize_to is not None:
        longest = max(orig_h, orig_w)

        if longest > resize_to:
            scale = resize_to / float(longest)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # store scale factors to rescale back later
            scale_x = orig_w / new_w
            scale_y = orig_h / new_h
        else:
            img_resized = img
    else:
        img_resized = img

    # Run Selective Search
    t0 = time.time()
    # rects = run_selective_search(img_resized, mode="fast", max_proposals=30) # this is faster but the other should be more accurate
    rects = run_selective_search(img_resized, mode="quality", max_proposals=30) # not sure if we want to keep 30
    dt = time.time() - t0

    print(f"[SS] {filename}: {len(rects)} proposals in {dt:.2f}s")

    # Rescale proposals back to original image size
    if resize_to is not None and (scale_x != 1.0 or scale_y != 1.0):
        rects = rects.astype(np.float32)
        rects[:, 0] *= scale_x
        rects[:, 1] *= scale_y
        rects[:, 2] *= scale_x
        rects[:, 3] *= scale_y
        rects = rects.astype(np.int32)

    # Save proposals
    np.savez_compressed(
        out_path,
        boxes_xywh=rects,
        orig_size=np.array([orig_h, orig_w]),
        filename=filename
    )

    print(f"[SAVE] {out_path}")


def main():
    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)

    all_images = splits["train"] + splits["test"]
    print(f"Total images to process: {len(all_images)}")

    for i, fname in enumerate(all_images):
        print(f"\n[{i+1}/{len(all_images)}] Processing {fname}")
        process_image(fname, resize_to=400)

if __name__ == "__main__":
    main()
