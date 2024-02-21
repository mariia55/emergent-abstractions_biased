import numpy as np
import pathlib
import h5py


from sklearn.model_selection import train_test_split

from img_dataset import shapes3d


def create_dataset(input_shape, path):

    dataset = h5py.File(path, 'r')
    data = dataset['images'][:]
    full_labels = dataset['labels'][:]

    #array= np.unique(full_labels[0], axis=1, return_index=False)
    trait_names_by_idx = ['floorHue', 'wallHue', 'color', 'scale', 'shape', 'orientation']

    floorHues = []
    wallHues = []
    colors = []
    scales =[]
    shapes = [] 
    orientations = []

    for elem in full_labels:
        floorHues.append(elem[0])
        wallHues.append(elem[1])
        colors.append(elem[2])
        scales.append(elem[3])
        shapes.append(elem[4])
        orientations.append(elem[5])

    all_floorHues = [0.,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_wallHues = [0.,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_colors = [0.,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_scales = [0.75 , 0.82142857, 0.89285714, 0.96428571, 1.03571429, 1.10714286, 1.17857143, 1.25]
    all_shapes = [0., 1., 2., 3.]
    all_orientations = [-30., -25.71428571, -21.42857143, -17.14285714, -12.85714286, -8.57142857, -4.28571429, 0., 
                        4.28571429,   8.57142857, 12.85714286,  17.14285714,  21.42857143,  25.71428571,  30.]

    colors_to_keep = [0., 0.3, 0.6, 0.9]
    scales_to_keep = [0.75, 0.89285714, 1.10714286, 1.25]
    shapes_to_keep = [0., 1., 2., 3.]

    indeces = []

    new_labels = []
    new_data = []
    for idx, sample in enumerate(full_labels):
        if (sample[2] in colors_to_keep) and (sample[3] in scales_to_keep) and (sample[4] in shapes_to_keep):
            new_labels.append(sample)
            indeces.append(idx)

    for idx in indeces:
        new_data.append(data[idx])
    
    # print("these are unique floorHues: ", np.unique(floorHues))
    # print("these are unique wallHues: ", np.unique(wallHues))
    # print("these are unique colors: ", np.unique(colors))
    # print("these are unique scales: ", np.unique(scales))
    # print("these are unique shapes: ", np.unique(shapes))
    # print("these are unique orientations: ", np.unique(orientations))
    print("both lengths: ", len(new_labels), len(new_labels))
    
# def fix_labels(labels):
#     new_labels = np.zeros(shape=labels.shape)
#     for label in labels:
create_dataset(0, '3dshapes/3dshapes.h5')