import torch 
import numpy as np
from torch.utils.data import Dataset

from dataset import DataSet
from load import load_data
from vision_module import feat_rep_vision_module


class shapes_dataset(Dataset):
    """
    This class uses given image, label and feature representation arrays to make a pytorch dataset out of them.
    The feature representations are left empty until 'generate_dataset()' is used to fill them.
    """
    def __init__(self, images = [], labels = [], feat_reps = [], transform = None):

        self.images = images  # array shape originally [480000,64,64,3], uint8 in range(256)
        self.feat_reps = feat_reps
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

def generate_dataset():
    """
    Function to create the feature representations and include them into the dataset
    """
    print("Starting to create the feature representation dataset")
    # load the trained model from save
    model = feat_rep_vision_module()
    try:
        model.load_state_dict(torch.load('./models/vision_module'), strict=False)
    except:
        raise ValueError('No trained vision module found in /models/vision_module')

    if torch.cuda.is_available():
        model.cuda()
    
    model.eval()

    data = torch.load('./dataset/complete_dataset')

    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size = 32, 
                                            shuffle = False,
                                            pin_memory = True)
    
    images = []
    labels = []
    feature_representations = []

    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            
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
    feat_rep_dataset_full = shapes_dataset(np.array(images), np.array(labels), np.array(feature_representations))
    torch.save(feat_rep_dataset_full, './dataset/complete_dataset')

    # and once as a much smaller dataset with the labels and feature representations but without the original images
    feat_rep_dataset_without_images = shapes_dataset(labels=np.array(labels), feat_reps=np.array(feature_representations))
    torch.save(feat_rep_dataset_without_images, './dataset/feat_rep_dataset')

    print("Saved feature representation dataset to /dataset twice, once with (complete_dataset) and once without images (feat_rep_datset)")
    return feat_rep_dataset_full, feat_rep_dataset_without_images

if __name__ == "__main__":
    try:
        
        complete_data = torch.load('./dataset/complete_dataset')
    
    except:
        input_shape = [3,64,64]
        
        full_data, labels_reg, full_labels = load_data(input_shape, normalize=False,
                                                                        subtract_mean=False,
                                                                        trait_weights=None,
                                                                        return_trait_weights=False,
                                                                        return_full_labels=True,
                                                                        datapath=None)
        
        complete_data = shapes_dataset(full_data, labels_reg)

        torch.save(complete_data, './dataset/complete_dataset')
    # generate the dataset with the feature representations first
    complete_dataset,_ = generate_dataset()

    # generate concept datasets for the communication game
    feat_rep_concept_dataset = DataSet(game_size=4, is_shapes3d=True, images=complete_dataset.feat_reps, labels=complete_dataset.labels)
    torch.save(feat_rep_concept_dataset, './dataset/feat_rep_concept_dataset_new')

    # also for the zero_shot dataset
    feat_rep_zero_concept_dataset = DataSet(game_size=4, zero_shot=True, is_shapes3d=True, images=complete_dataset.feat_reps, labels=complete_dataset.labels)
    torch.save(feat_rep_zero_concept_dataset, './dataset/feat_rep_zero_concept_dataset_new')
