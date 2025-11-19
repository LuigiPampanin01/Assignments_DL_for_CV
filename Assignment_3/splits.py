from pathlib import Path
import json
import random

# Dataset folder
DATA_ROOT = Path("/dtu/datasets1/02516/potholes")
IMG_DIR = DATA_ROOT / "images"
ANN_DIR = DATA_ROOT / "annotations"

# Output splits file
OUT_PATH = Path("splits.json") # ofc run it from the folder where you want to save it

def main():
    images = sorted(IMG_DIR.glob("*.png"))
    print(f"Found {len(images)} images in {IMG_DIR}")

    valid_filenames = []
    missing_xml = 0

    for img_path in images:
        xml_name = img_path.name.replace(".png", ".xml")
        xml_path = ANN_DIR / xml_name
        if xml_path.exists():
            valid_filenames.append(img_path.name)
        else:
            print(f"[WARN] Missing annotation for {img_path.name}")
            missing_xml += 1

    print(f"Images with annotation: {len(filenames)}")
    print(f"Images without annotation: {missing_xml}")

    # shuffle the dataset for randomness
    random.seed(42)
    random.shuffle(valid_filenames)

    n_total = len(valid_filenames)

    # Compute split sizes: 70% train, 15% val, 15% test
    # or 70, 20 and 10
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val  # remaining samples

    # Create the splits
    train_files = valid_filenames[:n_train]
    val_files   = valid_filenames[n_train : n_train + n_val]
    test_files  = valid_filenames[n_train + n_val :]

    print("\n===== SPLIT SIZES =====")
    print(f"Train: {len(train_files)}")
    print(f"Val  : {len(val_files)}")
    print(f"Test : {len(test_files)}")

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    with open(OUT_PATH, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\nSaved 70-15-15 splits to: {OUT_PATH}")

if __name__ == "__main__":
    main()
