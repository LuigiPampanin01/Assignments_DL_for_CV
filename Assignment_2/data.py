from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Sampler
import numpy as np

class FrameImageDataset(torch.utils.data.Dataset): # A dataset of individual, independent frames (images). To be used for aggregation of per-frame models.
    def __init__(self, 
    root_dir='/work3/ppar/data/ucf101',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label





class FrameVideoDataset(torch.utils.data.Dataset): # does not load or decode the .avi videos themselves. Instead, it uses the video file names (from /videos/.../*.avi) only as references to locate pre-extracted frame folders.+
    # A dataset of entire videos, each represented by multiple frames. Sequence (list or tensor) of frames + label.
    def __init__(self, 
    root_dir = '', 
    split = '', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx): # core method, called every time the DataLoader fetches a batch. Loads and returns one video sample (frames + label) given its index. 
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


class FrameImageDatasetTwoStream(torch.utils.data.Dataset): # A dataset of individual, independent frames (images). To be used for aggregation of per-frame models.
    def __init__(self, root_dir='', split='train', transform=None, transform2=None):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.flow_paths = sorted(glob(f'{root_dir}/flows/{split}/*/*/*.npy'))
        
        # Truncate to the shorter list
        if len(self.frame_paths) == 0:
            print("WARNING: No frame files found! Check your directory structure.")
        if len(self.flow_paths) == 0:
            print("WARNING: No flow files found! Check your directory structure.")

        min_len = min(len(self.frame_paths), len(self.flow_paths))
        self.frame_paths = self.frame_paths[:min_len]
        self.flow_paths = self.flow_paths[:min_len]
        
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.transform2 = transform2
        self.n_sampled_frames = 9
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        flow_paths = self.flow_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_name_flow = flow_paths.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        video_meta_flow = self._get_meta('video_name', video_name_flow)
        flow_frames_dir = os.path.dirname(flow_paths) 
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        flow_stack = self.load_frames(flow_frames_dir)

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)


        return frame, label, flow_stack

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            # Construct the filename for the .npy file
            frame_file = os.path.join(frames_dir, f"flow_{i}_{i+1}.npy")
            
            # Load the NumPy array from the file
            flow_data = np.load(frame_file)
            
            # Check the shape of the loaded data
            # print(f"Shape of loaded .npy file: {flow_data.shape}")
            
            # If the shape is (H, W, 2), transpose it to (2, H, W)
            if flow_data.shape[-1] == 2:
                flow_data = np.transpose(flow_data, (2, 0, 1))
                
            # Convert the NumPy array to a PyTorch tensor
            # Ensure the data type is float32 for most models
            flow_tensor = torch.from_numpy(flow_data).float()
            
            # Remove the call to self.transform2 here.
            # Any other necessary transformations (e.g., normalization)
            # should be handled differently or as part of the model's
            # forward pass, as the data is already a tensor.
            
            frames.append(flow_tensor)

        flow_stack = torch.stack(frames, dim=0)
        return flow_stack





if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = 'ucf10'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]



