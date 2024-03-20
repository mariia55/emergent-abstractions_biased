from torch.utils.data import Dataset
import numpy as np
import itertools
import torch.nn.functional as F
import random
import torch
import h5py

class shapes3d(Dataset):
    """This class uses given image and label arrays or tensors to make a pytorch dataset out of them"""
    def __init__(self, images, labels, transform = None, game_size = 4):
        self.images = images  # array shape originally [480000,64,64,3], uint8 in range(256)
        self.labels = labels  # array shape originally [480000,6], float64
        self.transform = transform # if transform should be applied
        self.game_size = game_size
        # fixed indeces for     color,    scale,      shape,      color and scale,    color and shape,      scale and shape,      color and scale and shape
        #self.fixed_vectors = [0,1,2,3], [4,5,6,7], [8,9,10,11], [0,1,2,3, 4,5,6,7], [0,1,2,3, 8,9,10,11], [4,5,6,7, 8,9,10,11], [0,1,2,3, 4,5,6,7, 8,9,10,11]

        # fixed values for    color,   scale,   shape, color and scale, color and shape, scale and shape, color and scale and shape
        self.fixed_vectors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0),        (1,0,1),            (0,1,1),       (1,1,1)]

        print("starting concepts")
        self.concepts = self.get_all_concepts()
        print("we are done with the concepts")
        self.properties_dim = [10, 10, 4, 4, 4, 15]

        self.dataset = self.get_datasets(split_ratio=(0.6, 0.2, 0.2))
    
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
                # print(concepts[-1])
        return concepts

    def get_distractors(self, concept_idx, context_condition):
        """
        Computes distractors.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        context = []

        # save fixed attribute indices in a list for later comparisons
        fixed_attr_indices = []
        for index, value in enumerate(fixed):
            if value == 1:
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
        for i, attr in enumerate(fixed):
            # if an attribute is fixed
            if attr == 1:
                # compare object with concept object
                if object[i] == concept_object[i]:
                    same_counter = same_counter +1
        # the number of shared attributes should match the number of fixed attributes
        if same_counter == sum(fixed):
            satisfied = True
        return satisfied

    def get_sample(self, concept_idx, context_condition):
        """
        Returns a full sample consisting of a set of target objects (target concept) 
        and a set of distractor objects (context) for a given concept condition.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        # sample target objects for given game size (if possible, get unique choices)
        try:
            target_objects = random.sample(all_target_objects, k=self.game_size)
        except ValueError:
            target_objects = random.choices(all_target_objects, k=self.game_size)
        # get all possible distractors for a given concept (for all possible context conditions)
        context = self.get_distractors(concept_idx, context_condition)
        context_sampled = self.sample_distractors(context, context_condition)
        print("this is context_condition: ", context_condition)
        print("this is target_objects: ", target_objects)
        print("this is fixed: ", fixed)
        print("this is corresponding context: ", context_sampled)
        # return target concept, context (distractor objects + context_condition) for each context
        return [target_objects, fixed], context_sampled

    def sample_distractors(self, context, context_condition):
        """
        Function for sampling the distractors from a specified context condition.
        """
        # sample distractor objects for given game size and the specified context condition
        #distractors = [dist_obj for dist_objs in context for dist_obj in dist_objs]
        context_new = []
        print("this is context: ", context)
        try: 
            context_new.append([random.sample(context, k=self.game_size), context_condition])
        except ValueError:
            context_new.append([random.choices(context, k=self.game_size), context_condition])
        return context_new
    
    def get_item(self, concept_idx, context_condition, include_concept=False):
        """
        Receives concept-context pairs and an encoding function.
        Returns encoded (sender_input, labels, receiver_input).
            sender_input: (sender_input_objects, sender_labels)
            labels: indices of target objects in the receiver_input
            receiver_input: receiver_input_objects
        The sender_input_objects and the receiver_input_objects are different objects sampled from the same concept 
        and context condition.
        """
        # use get_sample() to get sampled target and distractor objects 
        # The concrete sampled objects can differ between sender and receiver.
        sender_concept, sender_context = self.get_sample(concept_idx, context_condition)
        receiver_concept, receiver_context = self.get_sample(concept_idx, context_condition)
        # : change such that sender input also includes fixed vectors (i.e. full concepts) and fixed vectors are only 
        # ignored in the sender architecture
        # : also do this for context conditions?
        # initalize sender and receiver input with target objects only
        if include_concept == True:
            raise NotImplementedError		

        # subset such that only target objects are presented to sender and receiver
        sender_targets = sender_concept[0]
        receiver_targets = receiver_concept[0]
        sender_input = [obj for obj in sender_targets]
        receiver_input = [obj for obj in receiver_targets]
        # append context objects
        # get context of relevant context condition
        for distractor_objects, context_cond in sender_context:
                if context_cond == context_condition:
                    # add distractor objects for the sender
                    for obj in distractor_objects:
                        sender_input.append(obj)
        for distractor_objects, context_cond in receiver_context:
            if context_cond == context_condition:
                # add distractor objects for the receiver
                for obj in distractor_objects:
                    receiver_input.append(obj)

        # shuffle receiver input and create (many-hot encoded) label
        random.shuffle(receiver_input)
        receiver_label = [idx for idx, obj in enumerate(receiver_input) if obj in receiver_targets]
        receiver_label = torch.Tensor(receiver_label).to(torch.int64) # .to(device=self.device)
        receiver_label = F.one_hot(receiver_label, num_classes=self.game_size*2).sum(dim=0).float()
        # ENCODE and return as TENSOR
        sender_input = torch.stack([self._many_hot_encoding(elem) for elem in sender_input])
        receiver_input = torch.stack([self._many_hot_encoding(elem) for elem in receiver_input])
        # output needs to have the structure sender_input, labels, receiver_input
        return sender_input, receiver_label, receiver_input
    
    def get_datasets(self, split_ratio, include_concept=False):
        """
        Creates the train, validation and test datasets based on the number of possible concepts.
        """
        if sum(split_ratio) != 1:
            raise ValueError

        train_ratio, val_ratio, test_ratio = split_ratio

        # Shuffle sender indices
        concept_indices = torch.randperm(len(self.concepts)).tolist()
        # Split is based on how many distinct concepts there are (regardless context conditions)
        ratio = int(len(self.concepts)*(train_ratio + val_ratio))
        concept_indices.sort()
        print("concept indeces", concept_indices[0])

        train_and_val = []
        print("Creating train_ds and val_ds...")
        for concept_idx in concept_indices[:ratio]:
            for _ in range(self.game_size):
                # for each concept, we consider all possible context conditions
                # i.e. 1 for generic concepts, and up to len(properties_dim) for more specific concepts
                nr_possible_contexts = sum(self.concepts[concept_idx][1]) ######################################## len instead of sum here?
                for context_condition in range(nr_possible_contexts):
                    train_and_val.append(self.get_item(concept_idx, context_condition, include_concept))

        # Calculating how many train
        train_samples = int(len(train_and_val)*(train_ratio/(train_ratio+val_ratio)))
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])
        # Save information about train dataset
        train.dimensions = self.properties_dim

        test = []
        print("\nCreating test_ds...")
        for concept_idx in concept_indices[ratio:]:
            for _ in range(self.game_size):
                nr_possible_contexts = sum(self.concepts[concept_idx][1]) ######################################## len instead of sum here?
                for context_condition in range(nr_possible_contexts):
                    test.append(self.get_item(concept_idx, context_condition, include_concept))

        return train, val, test  
    
    def _many_hot_encoding(self, input_list):
        """
        Outputs a binary one dim vector
        """
        output = torch.zeros([sum(self.properties_dim)])#.to(device=self.device)
        start = 0

        for elem, dim in zip(input_list, self.properties_dim):
            output[start + elem] = 1
            start += dim
            
        return output

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
        
        attribute_dict = {
        0: [0, 0, 0],
        1: [0,0,1],
        2: [0,0,2],
        3: [0,0,3],
        4: [0,1,0],
        5: [0,1,1],
        6: [0,1,2],
        7: [0,1,3],
        8: [0,2,0],
        9: [0,2,1],
        10: [0,2,2],
        11: [0,2,3],
        12: [0,3,0],
        13: [0,3,1],
        14: [0,3,2],
        15: [0,3,3],
        16: [1,0,0],
        17: [1,0,1],
        18: [1,0,2],
        19: [1,0,3],
        20: [1,1,0],
        21: [1,1,1],
        22: [1,1,2],
        23: [1,1,3],
        24: [1,2,0],
        25: [1,2,1],
        26: [1,2,2],
        27: [1,2,3],
        28: [1,3,0],
        29: [1,3,1],
        30: [1,3,2],
        31: [1,3,3],
        32: [2,0,0],
        33: [2,0,1],
        34: [2,0,2],
        35: [2,0,3],
        36: [2,1,0],
        37: [2,1,1],
        38: [2,1,2],
        39: [2,1,3],
        40: [2,2,0],
        41: [2,2,1],
        42: [2,2,2],
        43: [2,2,3],
        44: [2,3,0],
        45: [2,3,1],
        46: [2,3,2],
        47: [2,3,3],
        48: [3,0,0],
        49: [3,0,1],
        50: [3,0,2],
        51: [3,0,3],
        52: [3,1,0],
        53: [3,1,1],
        54: [3,1,2],
        55: [3,1,3],
        56: [3,2,0],
        57: [3,2,1],
        58: [3,2,2],
        59: [3,2,3],
        60: [3,3,0],
        61: [3,3,1],
        62: [3,3,2],
        63: [3,3,3]}

        indeces = np.argmax(self.labels, axis = 1)
        unhottified = []
        for index in indeces:
            unhottified.append(attribute_dict[index])

        return unhottified