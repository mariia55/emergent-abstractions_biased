from torch.utils.data import Dataset
import h5py

class shapes3d(Dataset):
    """This class uses given image and label arrays or tensors to make a pytorch dataset out of them"""
    def __init__(self, images, labels, transform = None):
        self.images = images  # array shape originally [480000,64,64,3], uint8 in range(256)
        self.labels = labels  # array shape originally [480000,6], float64
        self.transform = transform # if transform should be applied
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return (img, label)