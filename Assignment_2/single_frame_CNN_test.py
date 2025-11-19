from collections import Counter
from datasets import FrameVideoDataset
from torchvision import transforms as T
import torch
from tqdm import tqdm
from single_frame_CNN import Network

# -------------- CONFIG ----------------
root_dir = 'Assignment_2/ufc10'
# root_dir = '/dtu/datasets1/02516/ucf101_noleakage'
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- LOAD MODEL ----------------
model = Network(num_classes=10).to(device)
model.load_state_dict(torch.load("Assignment_2/model_best_single_frame.pth", map_location=device))
model.eval()

# -------------- LOAD VIDEO DATASET ----------------
video_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=False)

correct = 0
total = len(video_dataset)

print("Starting evaluation")
# -------------- EVALUATE ----------------
with torch.no_grad():
    for i in tqdm(range(total), desc="Video-level evaluation"):
        frames, label = video_dataset[i]
        label = torch.tensor(label).to(device)

        frame_preds = []

        for frame in frames:
            frame = frame.unsqueeze(0).to(device)  # [1,3,H,W]
            output = model(frame)
            pred = output.argmax(1).item()
            frame_preds.append(pred)

        # Majority voting
        majority_class = Counter(frame_preds).most_common(1)[0][0]

        if majority_class == label.item():
            correct += 1

accuracy = correct / total * 100
print(f"\n🎬 Video-level Accuracy (Majority Voting): {accuracy:.2f}%")
