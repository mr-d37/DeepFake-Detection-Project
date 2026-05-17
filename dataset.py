import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


class DeepFakeDataset(Dataset):
    """
    PyTorch Dataset class for loading sequences of face frames.

    Each sample consists of a sequence of frames from a single video,
    along with a label (0 for real, 1 for fake).
    """

    def __init__(self, real_dir, fake_dir, num_frames=40, transform=None):
        """
        Initialize the dataset.

        Args:
            real_dir: Directory containing real video face extractions
            fake_dir: Directory containing fake video face extractions
            num_frames: Number of frames to load per video
            transform: Optional transform to apply to each frame
        """
        self.num_frames = num_frames
        self.transform = transform

        self.samples = []

        real_video_dirs = self._get_video_dirs(real_dir)
        for video_dir in real_video_dirs:
            self.samples.append((video_dir, 0))

        fake_video_dirs = self._get_video_dirs(fake_dir)
        for video_dir in fake_video_dirs:
            self.samples.append((video_dir, 1))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _get_video_dirs(self, base_dir):
        """Get all video directories in the base directory."""
        video_dirs = []
        base_path = Path(base_dir)

        if not base_path.exists():
            return video_dirs

        for item in base_path.iterdir():
            if item.is_dir():
                frame_files = list(item.glob("*.jpg")) + list(item.glob("*.png"))
                if len(frame_files) > 0:
                    video_dirs.append(str(item))

        return video_dirs

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (frames tensor, label)
            frames: Tensor of shape (num_frames, 3, 224, 224)
            label: 0 for real, 1 for fake
        """
        video_dir, label = self.samples[idx]

        frame_files = sorted(Path(video_dir).glob("*.jpg"))
        if len(frame_files) == 0:
            frame_files = sorted(Path(video_dir).glob("*.png"))

        if len(frame_files) == 0:
            raise ValueError(f"No frame files found in {video_dir}")

        if len(frame_files) >= self.num_frames:
            indices = list(range(len(frame_files)))
            selected_indices = indices[::len(indices) // self.num_frames][:self.num_frames]
            selected_frames = [frame_files[i] for i in selected_indices]
        else:
            selected_frames = []
            for i in range(self.num_frames):
                frame_idx = i % len(frame_files)
                selected_frames.append(frame_files[frame_idx])

        frames = []
        for frame_path in selected_frames:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        frames_tensor = torch.stack(frames)

        return frames_tensor, label


class VideoInferenceDataset(Dataset):
    """
    Dataset class for inference on new videos.
    Assumes faces have already been extracted using face_extractor.py
    """

    def __init__(self, video_dir, num_frames=40, transform=None):
        """
        Args:
            video_dir: Directory containing extracted face frames
            num_frames: Number of frames to load
            transform: Optional transform to apply to each frame
        """
        self.num_frames = num_frames
        self.video_dir = video_dir
        self.transform = transform

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        frame_files = sorted(Path(video_dir).glob("*.jpg"))
        if len(frame_files) == 0:
            frame_files = sorted(Path(video_dir).glob("*.png"))

        if len(frame_files) >= self.num_frames:
            indices = list(range(len(frame_files)))
            self.frame_files = [frame_files[i] for i in indices[::len(indices) // self.num_frames][:self.num_frames]]
        else:
            self.frame_files = []
            for i in range(self.num_frames):
                frame_idx = i % len(frame_files) if len(frame_files) > 0 else 0
                self.frame_files.append(frame_files[frame_idx])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        for frame_path in self.frame_files:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        frames_tensor = torch.stack(frames)
        return frames_tensor