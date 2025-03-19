import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from glob import glob
from classes.VideoDataset import VideoDataset

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

video_paths = glob("dataset/videos/*.mp4")  # Replace with your dataset path
labels = [int(os.path.basename(path).split('_')[0]) for path in video_paths]  # Example label extraction

dataset = VideoDataset(video_paths, labels, num_segments=3, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
