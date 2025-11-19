"""
Visualize Selective Search proposals for a single image.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image


# =============================
# Paths
# =============================
DATA_ROOT = Path("/dtu/datasets1/02516/potholes")
IMG_DIR = DATA_ROOT / "images"
PROPOSAL_DIR = Path.home() / "IDLCV_Project4" / "proposals_ss"
OUT_DIR = Path.home() / "IDLCV_Project4" / "vis_proposals"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_proposals(filename: str, max_boxes: int = 200):
    """
    Visualize the first `max_boxes` Selective Search proposals for one image.

    Parameters:
    - filename  : name of the pothole image (e.g., "potholes10.png")
    - max_boxes : how many proposals to draw for visualization
    """

    # Load proposals
    proposal_path = PROPOSAL_DIR / f"{Path(filename).stem}_ss.npz"
    data = np.load(proposal_path)

    boxes = data["boxes_xywh"]
    print(f"Total proposals: {boxes.shape[0]}")

    # Limit boxes for visualization
    boxes = boxes[:max_boxes]

    # Load image
    img_path = IMG_DIR / filename
    img = Image.open(img_path).convert("RGB")

    # Plot image
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    # Draw proposals
    for (x, y, w, h) in boxes:
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
            alpha=0.5,
        )
        ax.add_patch(rect)

    plt.title(f"{filename} — showing {max_boxes} proposals")
    plt.axis("off")

    # Save visualization
    out_path = OUT_DIR / f"props_{Path(filename).stem}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization → {out_path}")


if __name__ == "__main__":
    # Example visualization
    filename = "potholes10.png"  # change as needed
    visualize_proposals(filename, max_boxes=200)
