from torch.utils.data import Dataset
import numpy as np
import itertools
import h5py

class shapes3d(Dataset):
    """This class uses given image and label arrays or tensors to make a pytorch dataset out of them"""
    def __init__(self, images, labels, transform = None, game_size = 0):
        self.images = images  # array shape originally [480000,64,64,3], uint8 in range(256)
        self.labels = labels  # array shape originally [480000,6], float64
        self.transform = transform # if transform should be applied
        # fixed indeces for     color,    scale,      shape,      color and scale,    color and shape,      scale and shape,      color and scale and shape
        self.fixed_vectors = [0,1,2,3], [4,5,6,7], [8,9,10,11], [0,1,2,3, 4,5,6,7], [0,1,2,3, 8,9,10,11], [4,5,6,7, 8,9,10,11], [0,1,2,3, 4,5,6,7, 8,9,10,11]

        #self.concept = self.get_all_concepts()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return (img, label)
    
    def get_all_concepts(self):
        all_objects = self.un_one_hottify()

        all_fixed_object_pairs = list(itertools.product(all_objects, self.fixed_vectors))

        concepts = list()
		# go through all concepts (i.e. fixed, objects pairs)
        for concept in all_fixed_object_pairs:
            # treat each fixed_object pair as a target concept once
            # e.g. target concept (_, _, 0) (i.e. fixed = (0,0,1) and objects e.g. (0,0,0), (1,0,0))
            fixed = concept[1]
            # go through all objects and check whether they satisfy the target concept (in this example have 0 as 3rd attribute)
            target_objects = list()
            for object in all_objects:
                if self.satisfies(object, concept):
                    if object not in target_objects:
                        target_objects.append(object)
            # concepts are tuples of fixed attributes and all target objects that satisfy the concept
            if (target_objects, fixed) not in concepts:
                concepts.append((target_objects, fixed))
                print(concepts[-1])
        return concepts

    def get_distractors(self, concept_idx, context_condition):
        """
        Computes distractors.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        context = []

        # save fixed attribute indices in a list for later comparisons
        fixed_attr_indices = []
        for index in fixed:
            fixed_attr_indices.append(index)

        # consider all objects as possible distractors
        poss_dist = self.un_one_hottify()

        for obj in poss_dist:
            # find out how many attributes are shared between the possible distractor object and the target concept
            # (by only comparing fixed attributes because only these are relevant for defining the context)
            shared = sum(1 for idx in fixed_attr_indices if obj[idx] == all_target_objects[0][idx])
            if shared == context_condition:
                context.append(obj)

        return context
    
    def satisfies(self, object, concept):
        """
        Checks whether an object satisfies a target concept, returns a boolean value.
        Concept consists of an object vector and a fixed vector tuple.
        """
        satisfied = False
        same_counter = 0
        concept_object, fixed = concept
        # an object satisfies a concept if fixed attributes are the same
        # go through attributes an check whether they are fixed
        for i in fixed:
            # compare object with concept object
            if object[i] == concept_object[i]:
                same_counter = same_counter +1
        # the number of shared attributes should match the number of fixed attributes
        if same_counter == len(fixed):
            satisfied = True
        return satisfied

    def un_one_hottify(self):
        attribute_dict = {
        0: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        1: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        2: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        3: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        4: [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        5: [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        7: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        8: [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        9: [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        10: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        11: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        12: [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        13: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        14: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        15: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        16: [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        17: [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        18: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        19: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        20: [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        21: [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        22: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        23: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        24: [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        25: [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        26: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        27: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        28: [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        29: [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        30: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        31: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        32: [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        33: [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        34: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        35: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        36: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        37: [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        38: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        39: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        40: [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        41: [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        42: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        43: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        44: [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        45: [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        46: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        47: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        48: [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        49: [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        50: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        51: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        52: [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        53: [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        54: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        55: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        56: [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        57: [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        58: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        59: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        60: [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        61: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        62: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        63: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]}
        
        indeces = np.argmax(self.labels, axis = 1)
        unhottified = []
        for index in indeces:
            unhottified.append(attribute_dict[index])

        return unhottified