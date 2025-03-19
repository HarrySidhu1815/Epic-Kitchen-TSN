import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_segments=3, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_segments = num_segments
        self.transform = transform

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames(self, frames):
        num_frames = len(frames)
        if num_frames < self.num_segments:
            return np.random.choice(frames, self.num_segments, replace=True)
        indices = np.linspace(0, num_frames - 1, self.num_segments, dtype=int)
        return [frames[i] for i in indices]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path)
        sampled_frames = self._sample_frames(frames)

        if self.transform:
            sampled_frames = [self.transform(frame) for frame in sampled_frames]

        return torch.stack(sampled_frames), torch.tensor(label, dtype=torch.long)
