import os
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_single(image_path, boxes):
    """Display image with bounding boxes."""

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Create plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw all bounding boxes
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.title(os.path.basename(image_path))
    plt.axis("off")
    plt.savefig(os.path.join("images", os.path.basename(image_path)))


def visualize(path):

    xlm_files = os.listdir(os.path.join(path, "annotations"))


    for xlm in xlm_files:

        filename, list_with_all_boxes = read_content(os.path.join(path, "annotations", xlm))

        image_path = os.path.join(path, "images", filename)

        visualize_single(image_path, list_with_all_boxes)


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes
    

if __name__=="__main__":

    path = "/dtu/datasets1/02516/potholes"
    filename, list_with_all_boxes = visualize(path)

    print(filename, list_with_all_boxes)