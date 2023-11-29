import numpy as np
from torch.utils.data import Dataset
import h5py

class shapes3d(Dataset):
    def __init__(self, path_to_dataset = '3dshapes/3dshapes.h5', transform = None):
        load_dataset = h5py.File(path_to_dataset, 'r')
        self.images = np.asarray(load_dataset['images'])  # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = np.asarray(load_dataset['labels'])  # array shape [480000,6], float64
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.images(index)
        label = self.labels(index)
        if self.transform:
            img = self.transform(img)
        return (img,label)