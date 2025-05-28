import torch 
import numpy as np
from torch.utils.data import Dataset

from dataset import DataSet
from load import load_data
from vision_module import feat_rep_vision_module
#from shapes_dataset import ShapesDataset

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

def generate_dataset():
    """
    Function to create the feature representations and include them into the dataset
    """
    print("Starting to create the feature representation dataset")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # load the trained model from save
    model = feat_rep_vision_module().to(device)
    model_path = './models/vision_module'
    dataset_path = 'dataset/complete_dataset'
    try:
        model.load_state_dict(torch.load(model_path), strict=False)
    except:
        raise ValueError(f'No trained vision module found in {model_path}. Please train the model first.')

    try:
        data = torch.load(dataset_path)
    except:
        raise ValueError(f'ShapesDataset not found in {dataset_path}. Please train the model first, which also creates the dataset.')

    model.to(device)
    model.eval()

    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=32,
                                              shuffle=False,
                                              pin_memory=True)
    
    images = []
    labels = []
    feature_representations = []

    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            input = input.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            
            feat_rep = model(input)

            images_flat = torch.flatten(input, start_dim=0, end_dim=0)
            labels_flat = torch.flatten(target, start_dim=0, end_dim=0)
            feat_rep_flat = torch.flatten(feat_rep, start_dim=0, end_dim=0)

            for image in images_flat:
                images.append(image.cpu().numpy())
            for label in labels_flat:
                labels.append(label.cpu().numpy())
            for feat_rep in feat_rep_flat:
                feature_representations.append(feat_rep.cpu().numpy())

    # for size reasons the dataset is saved twice, 
    # once as the full dataset now including the feature representations
    feat_rep_dataset_full = ShapesDataset(np.array(images), np.array(labels), np.array(feature_representations))
    torch.save(feat_rep_dataset_full, dataset_path + '_feat_rep', pickle_protocol=4)

    # and once as a much smaller dataset with the labels and feature representations but without the original images
    feat_rep_dataset_without_images = ShapesDataset(labels=np.array(labels), feat_reps=np.array(feature_representations))
    torch.save(feat_rep_dataset_without_images, dataset_path + '_feat_rep_no_images', pickle_protocol=4)

    print(f"Feature representations saved to {dataset_path + '_feat_rep'} and {dataset_path + '_feat_rep_no_images'}")
    return feat_rep_dataset_full, feat_rep_dataset_without_images

def load_or_create_dataset(dataset_path, device='cpu', game_size = 4):
    """
    Loads the image representation dataset if it exists, otherwise creates it
    dataset_path: str, path to the dataset file
    device: str, device to load the dataset on

    returns: dataset.Dataset object
    """
    try:
        data_set = torch.load(dataset_path)
    except:
        print("Feature representations not found, creating it instead...")
        complete_dataset, _ = generate_dataset()
        data_set = DataSet(game_size=game_size,
                           is_shapes3d=True,
                           images=complete_dataset.feat_reps,
                           labels=complete_dataset.labels,
                           device=device)
        torch.save(data_set, dataset_path, pickle_protocol=4)

    return data_set

if __name__ == "__main__":
    feat_rep_dataset_path = 'dataset/complete_dataset' + '_feat_rep'
    try:
        complete_dataset = torch.load(feat_rep_dataset_path)
    except:
        print('Feature representations not found, creating it instead...')
        complete_dataset, _ = generate_dataset()

    # generate concept datasets for the communication game
    feat_rep_concept_dataset = DataSet(game_size=10, is_shapes3d=True, images=complete_dataset.feat_reps, labels=complete_dataset.labels, device='mps')
    torch.save(feat_rep_concept_dataset, './dataset/feat_rep_concept_dataset.pt', pickle_protocol=4)

    # also for the zero_shot dataset
    feat_rep_zero_concept_dataset = DataSet(game_size=10, zero_shot=True, zero_shot_test="generic", is_shapes3d=True, images=complete_dataset.feat_reps, labels=complete_dataset.labels, device='mps')
    torch.save(feat_rep_zero_concept_dataset, './dataset/feat_rep_zero_concept_dataset_generic.pt', pickle_protocol=4)

    feat_rep_zero_concept_dataset = DataSet(game_size=10, zero_shot=True, zero_shot_test="specific", is_shapes3d=True, images=complete_dataset.feat_reps, labels=complete_dataset.labels, device='mps')
    torch.save(feat_rep_zero_concept_dataset, './dataset/feat_rep_zero_concept_dataset_specific.pt', pickle_protocol=4)
