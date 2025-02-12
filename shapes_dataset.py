import torch 
import numpy as np
from torch.utils.data import Dataset

from dataset import DataSet
from load import load_data
from vision_module import feat_rep_vision_module


class ShapesDataset(Dataset):
    """
    This class uses given image, label and feature representation arrays to make a pytorch dataset out of them.
    The feature representations are left empty until 'generate_dataset()' is used to fill them.
    """
    def __init__(self, images=None, labels=None, feat_reps=None, transform=None):
        if images is None and labels is None:
            raise ValueError('No images or labels given')
        self.images = images  # array shape originally [480000,64,64,3], uint8 in range(256)
        self.feat_reps = feat_reps
        self.labels = labels  # array shape originally [480000,6], float64
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label