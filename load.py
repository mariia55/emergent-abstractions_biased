import numpy as np
import pathlib
import h5py

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from img_dataset import shapes3d

# code taken from / inspired by https://github.com/XeniaOhmer/language_perception_communication_games/

def coarse_generic_relational_labels(label_map):
    num_classes = np.max([int(k) for k in label_map.keys()]) + 1
    output_dicts = [{} for _ in range(len(label_map[0]))]
    for i in range(num_classes):
        traits_i = label_map[i]
        tmp_labels = [np.zeros(num_classes) for _ in range(len(traits_i))]
        for j in range(num_classes):
            if i == j:
                continue
            traits_j = label_map[j]
            for ii in range(len(traits_j)):
                if traits_i[ii] == traits_j[ii]:
                    tmp_labels[ii][j] = 1

        for j in range(len(tmp_labels)):
            tmp_labels[j] /= np.sum(tmp_labels[j])
            output_dicts[j][i] = tmp_labels[j]

    output_dicts = tuple(output_dicts)

    return output_dicts

def get_shape_color_labels(full_labels,
                           trait_idxs=(2, 3, 4)):
    possible_values = [[] for _ in range(len(trait_idxs))]
    trait_names_by_idx = ['floorHue', 'wallHue', 'color', 'scale', 'shape', 'orientation']

    # default trait_idxs set to (2,3,4) corresponding to color, scale, and shape
    extracted_traits = [tuple(entry) for entry in list(full_labels[:, trait_idxs])]

    for tup in extracted_traits:
        for (idx, entry) in enumerate(tup):
            possible_values[idx].append(entry)
    for (idx, p) in enumerate(possible_values):
        possible_values[idx] = sorted(set(p))

    # since there were only 4 possible shapes, we extracted 4 approximately equally spaced
    # values from the other two traits. balance_type == 2 was used for our experiments. The first
    # list in idxes_to_keep selects values for color and the second list selects values for
    # the object scale, based on the configuration set by the extracted_traits variable

    idxes_to_keep = [[0, 2, 4, 8], [0, 3, 5, 7]]
    values_to_keep = [[], []]

    for idx in [0, 1]:
        for val_idx in idxes_to_keep[idx]:
            values_to_keep[idx].append(possible_values[idx][val_idx])
    filtered_traits = []
    keeper_idxs = []
    for (idx, traits) in enumerate(extracted_traits):
        if traits[0] in values_to_keep[0] and traits[1] in values_to_keep[1]:
            filtered_traits.append(traits)
            keeper_idxs.append(idx)
    
    extracted_traits = filtered_traits

    trait_names = [trait_names_by_idx[i] for i in trait_idxs]
    unique_traits = sorted(set(extracted_traits))
    labels = np.zeros((len(extracted_traits), len(unique_traits)))

    # these dictionaries are used to convert between indices for one-hot target vectors
    # and the corresponding trait combination that that entry represents, which defines the class
    # composition of the classification problem
    label2trait_map = dict()
    trait2label_map = dict()

    for (i, traits) in enumerate(unique_traits):
        trait2label_map[traits] = i
        label2trait_map[i] = traits

    # use coarse labels
    labels_template = coarse_generic_relational_labels(label2trait_map)

    test = coarse_generic_relational_labels(label2trait_map)
    relational_labels = dict()
    test_relational_labels = dict()
    
    # calculating for individual traits
    for (i, k) in enumerate(trait_names):
        relational_labels[k] = labels_template[i]
        test_relational_labels[k] = test[i]

    # calculating for dual traits

    trait_weights = dict()
    trait_weights['color-shape'] = [0.5, 0.5]
    trait_weights['color-size'] = [0.5, 0.5]
    trait_weights['shape-size'] = [0.5, 0.5]

    relational_labels['color-shape'] = dict()
    relational_labels['color-size'] = dict()
    relational_labels['shape-size'] = dict()

    for idx in labels_template[0].keys():
        relational_labels['color-shape'][idx] = 0
        relational_labels['color-size'][idx] = 0
        relational_labels['shape-size'][idx] = 0
        for (i, k) in enumerate(trait_names):
            if k == 'color':
                relational_labels['color-shape'][idx] += trait_weights['color-shape'][0] * labels_template[i][idx]
                relational_labels['color-size'][idx] += trait_weights['color-shape'][0] * labels_template[i][idx]
            elif k == 'scale':
                relational_labels['shape-size'][idx] += trait_weights['shape-size'][1] * labels_template[i][idx]
                relational_labels['color-size'][idx] += trait_weights['color-size'][1] * labels_template[i][idx]
            elif k == 'shape':
                relational_labels['shape-size'][idx] += trait_weights['shape-size'][0] * labels_template[i][idx]
                relational_labels['color-shape'][idx] += trait_weights['color-shape'][1] * labels_template[i][idx]

    # calculating for all traits
    relational_labels['all'] = dict()
    for k in labels_template[0].keys():
        relational_labels['all'][k] = 0
        for lab in labels_template:
            relational_labels['all'][k] += 1 / len(labels_template) * lab[k]

    # generating one-hot labels
    for (i, traits) in enumerate(extracted_traits):
        labels[i, trait2label_map[traits]] = 1
    return labels, relational_labels, keeper_idxs, trait_weights


def load_data(input_shape, normalize=True,
              subtract_mean=True,
              trait_weights=None,
              return_trait_weights=False,
              return_full_labels=False,
              datapath=None):
    assert return_trait_weights + return_full_labels < 2, 'only can return one of trait_weights or full_labels'

    if datapath is None:
        data_path = '3dshapes/3dshapes.h5'
    else:
        data_path = datapath
    parent_dir = str(pathlib.Path().absolute()).split('/')[-1]
    if parent_dir == 'SimilarityGames':
        data_path = data_path[3:]
    try:
        dataset = h5py.File(data_path, 'r')
    except:
        raise ValueError("h5 file was not found at given path: ", data_path, " please download the h5 file for the dataset and give the correct path to it's location")
    
    data = dataset['images'][:]
    full_labels = dataset['labels'][:]
    labels_reg, labels_relational, keeper_idxs, trait_weights = get_shape_color_labels(full_labels)

    # chooses one of 3 variables to return as the meta variable - note, only one of the boolean return
    # variables should be set to True
    if return_full_labels:
        meta = full_labels
    elif return_trait_weights:
        meta = trait_weights
    else:
        meta = labels_relational

    if keeper_idxs is not None:
        data = np.array([data[idx] for idx in keeper_idxs])

    (train_data, test_data, train_labels, test_labels) = train_test_split(data, labels_reg,
                                                                          test_size=0.2) # 0.2 to get 0.2 to 0.8 train and test split
    
    (train_data, val_data, train_labels, val_labels) = train_test_split(train_data, train_labels,
                                                                          test_size=0.25) # 0.25 from the 0.8 training data to achieve overall split of 0.6 train, 0.2 val, 0.2 test

    train_data = train_data.reshape((train_data.shape[0],
                                        input_shape[0], input_shape[1], input_shape[2]))
    test_data = test_data.reshape((test_data.shape[0],
                                    input_shape[0], input_shape[1], input_shape[2]))
    val_data = val_data.reshape((val_data.shape[0],
                                    input_shape[0], input_shape[1], input_shape[2]))
    full_data = data.reshape((data.shape[0],
                                    input_shape[0], input_shape[1], input_shape[2]))

    if normalize:
        train_data = train_data.astype("float32") / 255.0
        test_data = test_data.astype("float32") / 255.0
        val_data = val_data.astype("float32") / 255.0
        full_data = full_data.astype("float32") / 255.0
    if subtract_mean:
        tmp_data = train_data.reshape(train_data.shape[1], -1)

        mean = np.mean(tmp_data, axis=1)
        train_data = train_data - mean
        test_data = test_data - mean
        val_data = val_data - mean
        full_data = full_data - mean

    if len(train_labels.shape) == 1 or train_labels.shape[1] == 1:
        le = LabelBinarizer()
        train_labels = le.fit_transform(train_labels)
        test_labels = le.transform(test_labels)
        val_labels = le.fit_transform(val_labels)
        target_names = [str(x) for x in le.classes_]
    else:
        target_names = []

    train_data = shapes3d(train_data, train_labels)
    test_data = shapes3d(test_data, test_labels)
    val_data = shapes3d(val_data, val_labels)
    complete_data = shapes3d(full_data, labels_reg)

    return train_data, val_data, test_data, target_names, meta, complete_data