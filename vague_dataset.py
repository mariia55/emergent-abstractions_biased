# code inspired by https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/dataset.py

import torch
import torch.nn.functional as F
import itertools
import random
from tqdm import tqdm

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)


class DataSet(torch.utils.data.Dataset):
    """
    This class provides the torch.Dataloader-loadable dataset.
    """

    def __init__(self, properties_dim=None, game_size=10, device=None, testing=False, zero_shot=False, zero_shot_test=None):
        """
		properties_dim: vector that defines how many attributes and features per attributes the dataset should contain, defaults to a 3x3x3 dataset
		game_size: integer that defines how many targets and distractors a game consists of
		"""
	    if properties_dim is None:
			properties_dim = [3, 3, 3]
		if device is None:
			device =  'cuda'
		if zero_shot_test is None:
			zero_shot_test = 'generic'

        super().__init__()

        self.properties_dim = properties_dim
        self.game_size = game_size
        self.device = device

        # get all concepts
        self.concepts = self.get_all_concepts()
        # get all objects
        self.all_objects = self._get_all_possible_objects(properties_dim)

        # generate dataset
        if not testing and not zero_shot:
            self.dataset = self.get_datasets(split_ratio=SPLIT)
        if zero_shot:
            self.dataset = self.get_zero_shot_datasets(
                split_ratio=SPLIT_ZERO_SHOT, test_cond=zero_shot_test
            )


    def __len__(self):
        """Returns the total amount of samples in dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns the i-th sample (and label?) given an index (idx)."""
        return self.dataset[idx]

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
        ratio = int(len(self.concepts) * (train_ratio + val_ratio))

        train_and_val = []
        print("Creating train_ds and val_ds...")
        for concept_idx in tqdm(concept_indices[:ratio]):
            for _ in range(self.game_size):
                # for each concept, we consider all possible context conditions
                # i.e. 1 for generic concepts, and up to len(properties_dim) for more specific concepts
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                for context_condition in range(nr_possible_contexts):
                    train_and_val.append(
                        self.get_item(
                            concept_idx,
                            context_condition,
                            self._normalized_encoding,
                            include_concept,
                        )
                    )

        # Calculating how many train
        train_samples = int(
            len(train_and_val) * (train_ratio / (train_ratio + val_ratio))
        )
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(
            train_and_val, [train_samples, val_samples]
        )
        # Save information about train dataset
        train.dimensions = self.properties_dim

        test = []
        print("\nCreating test_ds...")
        for concept_idx in tqdm(concept_indices[ratio:]):
            for _ in range(self.game_size):
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                for context_condition in range(nr_possible_contexts):
                    test.append(
                        self.get_item(
                            concept_idx,
                            context_condition,
                            self._normalized_encoding,
                            include_concept,
                        )
                    )

        return train, val, test

    def get_zero_shot_datasets(
        self, split_ratio, test_cond="generic", include_concept=False
    ):
        """
        Note: Generates train, val and test data.
            Test and training set contain different concepts. There are two possible datasets:
            1) 'generic': train on more specific concepts, test on most generic concepts
            2) 'specific': train on more generic concepts, test on most specific concepts
        :param split_ratio Tuple of ratios (train, val) of the samples should be in the training and validation sets.
        """

        if sum(split_ratio) != 1:
            raise ValueError

            # For each category, one attribute will be chosen for zero shot
            # The attributes will be taken from a random object
            # zero_shot_object = pd.Series([0 for _ in self.properties_dim])  # self.objects.sample().iloc[0]

            # split ratio applies only to train and validation datasets - size of test dataset depends on available concepts
        train_ratio, val_ratio = split_ratio

        train_and_val = []
        test = []

        print("Creating train_ds, val_ds and test_ds...")
        for concept_idx in tqdm(range(len(self.concepts))):
            for _ in range(self.game_size):
                # for each concept, we consider all possible context conditions
                # i.e. 1 for generic concepts, and up to len(properties_dim) for more specific concepts
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                # print("nr poss cont", nr_possible_contexts)
                for context_condition in range(nr_possible_contexts):
                    # 1) 'generic'
                    if test_cond == "generic":
                        # test dataset only contains most generic concepts
                        if nr_possible_contexts == 1:
                            test.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._normalized_encoding,
                                    include_concept,
                                )
                            )
                        else:
                            train_and_val.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._normalized_encoding,
                                    include_concept,
                                )
                            )

                            # 2) 'specific'
                    if test_cond == "specific":
                        # test dataset only contains most specific concepts
                        if nr_possible_contexts == len(self.properties_dim):
                            test.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._normalized_encoding,
                                    include_concept,
                                )
                            )
                        else:
                            train_and_val.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._normalized_encoding,
                                    include_concept,
                                )
                            )

                            # Train val split
        train_samples = int(len(train_and_val) * train_ratio)
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(
            train_and_val, [train_samples, val_samples]
        )

        # Save information about train dataset
        train.dimensions = self.properties_dim
        print("Length of train and validation datasets:", len(train), "/", len(val))
        print("Length of test dataset:", len(test))

        return train, val, test

    def get_item(
        self, concept_idx, context_condition, encoding_func, include_concept=False
    ):
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
        receiver_concept, receiver_context = self.get_sample(
            concept_idx, context_condition
        )
        # TODO: change such that sender input also includes fixed vectors (i.e. full concepts) and fixed vectors are only
        # ignored in the sender architecture
        # NOTE: also do this for context conditions?
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
        # sender input does not need to be shuffled - that way I don't need labels either
        # random.shuffle(sender_input)
        # sender_label = [idx for idx, obj in enumerate(sender_input) if obj in sender_targets]
        # sender_label = torch.Tensor(sender_label).to(torch.int64)
        # sender_label = F.one_hot(sender_label, num_classes=self.game_size*2).sum(dim=0).float()
        # shuffle receiver input and create (many-hot encoded) label
        random.shuffle(receiver_input)
        receiver_label = [
            idx for idx, obj in enumerate(receiver_input) if obj in receiver_targets
        ]
        receiver_label = (
            torch.Tensor(receiver_label).to(torch.int64).to(device=self.device)
        )
        receiver_label = (
            F.one_hot(receiver_label, num_classes=self.game_size * 2).sum(dim=0).float()
        )
        # ENCODE and return as TENSOR
        sender_input = torch.stack([encoding_func(elem) for elem in sender_input])
        receiver_input = torch.stack([encoding_func(elem) for elem in receiver_input])
        # output needs to have the structure sender_input, labels, receiver_input
        # return torch.cat([sender_input, sender_label]), receiver_label, receiver_input
        return sender_input, receiver_label, receiver_input

    def sample_attribute(self, min_value, max_value):
        """
        This function samples a random number within the range [min_value, max_value].
        """
        sample_attribute = min_value + torch.rand(1) * (max_value - min_value)
        return sample_attribute

    def sample_target(self, concept_ranges):
        """ This function samples a target object based on ranges for each attribute.
        It iterates over each range, calling 'sample_attribute' for each.
        'r' is a tuple representing the range for one attribute, with 'r[0]' being the lower bound and 'r[1]' the upper bound.
        """
        sample_target = [self.sample_attribute(r[0], r[1]).item() for r in concept_ranges]
        return sample_target

    def sample_distractor(concept_ranges, context='fine', overlap=0.1):
        """
        This function generates a distractor object based on the context condition and desired overlap.
        """
        distractor = []
        for r in concept_ranges:
            if context == 'fine':
                # Ensure some overlap with the target range for a 'fine' context.
                min_value = max(0, r[0] - overlap)
                max_value = min(1, r[1] + overlap)
            else:
                # Ensure no overlap (separation) for a 'coarse' context.
                if torch.rand(1).item() > 0.5:
                    # Choose to separate below or above the target range randomly.
                    min_value = max(0, r[0] - 1)
                    max_value = r[0] - overlap
                else:
                    min_value = r[1] + overlap
                    max_value = min(1, r[1] + 1)
            # Sample a value within the calculated bounds for the distractor.
            distractor.append(self.sample_attribute(min_value, max_value).item())
        return distractor
    
    def get_sample(self, concept_idx, context_condition):
        """
        Returns a full sample consisting of a set of target objects (target concept)
        and a set of distractor objects (context) for a given concept condition.
        """
        # Get the concept range for the current concept index
        concept_ranges = self.concepts[concept_idx]

        # Sample a target object using the concept ranges
        target_object = self.sample_target(concept_ranges)

        # Initialize an empty list to hold the distractors
        distractors = []

        # Define the number of distractors you want to generate
        num_distractors = self.game_size - 1

        # Generate the required number of distractor objects
        for _ in range(num_distractors):
            distractor = self.sample_distractor(concept_ranges, context=context_condition)
            distractors.append(distractor)

        # Return the target object and the list of distractor objects
        return target_object, distractors

           
    def get_all_concepts(self):
        """
        Returns all possible concepts for a given dataset size with continuous attributes.
        Each concept is represented as a range of values for each attribute.
        """
        all_objects = self._get_all_possible_objects(self.properties_dim)
        concepts = []

        # Define the width of the range around the attribute value that is considered part of the concept.
        range_width = 0.1
        
        # Generate concepts
        for obj in all_objects:
            concept_range = [(max(0, value - range_width), min(1, value + range_width)) for value in obj]
            
            # Find objects that fit into this concept
            concept_objects = [o for o in all_objects if self.satisfies(o, concept_range)]
            
            # Add concept (range and objects that fit into it) to the concepts list
            if concept_objects:
                concepts.append((concept_objects, concept_range))

        return concepts
    
    @staticmethod
    def satisfies(object_attributes, concept_ranges):
        """
        Checks if an object fits within the concept ranges.
        It returns True if all attributes ('attr') of the object fall within the corresponding ranges ('r').
        """
        return all(r[0] <= attr <= r[1] for attr, r in zip(object_attributes, concept_ranges))


    @staticmethod
    def get_all_objects_for_a_concept(properties_dim, feature_ranges):
        """
        Generates all possible objects for a concept given the range of each attribute for a dataset with continuous attributes.
        
        properties_dim: A list of the number of divisions or samples within the range for each attribute.
        feature_ranges: A list of tuples where each tuple consists of (min_value, max_value) for each attribute.
        
        Returns a list of concept objects where each object is a tuple of values within the specified ranges.
        """
        all_objects = []
        
        # For each attribute's range, generate a set of points.
        for dim, (min_val, max_val) in zip(properties_dim, feature_ranges):
            # Assumes an equal distribution of points within the range
            all_objects.append(torch.linspace(min_val, max_val, steps=dim).tolist())
            
        concept_objects = list(itertools.product(*all_objects))

        return concept_objects

    @staticmethod
    def _get_all_possible_objects(properties_dim):
        """
        Returns all possible combinations of attribute-feature values as a list of tuples.
        """
        # Create a list of torch tensors, each with randomly sampled values between 0 and 1
        continuous_dims = [torch.rand(dim) for dim in properties_dim]
    
        # Convert each tensor to a list and then create a Cartesian product of these lists
        all_objects = list(itertools.product(*[dim.tolist() for dim in continuous_dims]))
    
        return all_objects 

    def _normalized_encoding(self, input_list):
        """
        This function outputs a normalized encoded tensor for a given input list where each element in the input list corresponds to an attribute of an object, 
        and the values are indices within their respective attribute's range. 
        """
        output = torch.tensor(input_list, device=self.device)
        # Add a small epsilon to avoid division by zero
        output_sum = output.sum().item()
        if output_sum == 0:
            output_sum = 1e-10  # small number to avoid division by zero
        # Normalizes the vector to have a sum of 1
        output = output / output_sum
        return output
    
    def create_game_instance(self, target_object, all_objects, context_condition):
        game_instance = {
            "target": target_object,
            "distractors": []
        }

        # Generate the required number of distractor objects
        for _ in range(self.game_size - 1):
            distractor = self.sample_distractor(all_objects, context=context_condition)
            game_instance["distractors"].append(distractor)

        return game_instance
    
    def create_game_instances(self, num_instances, context_condition):
        game_instances = []

        for _ in range(num_instances):
            # Sample a concept for the target
            concept_idx = np.random.randint(len(self.concepts))
            concept_ranges = self.concepts[concept_idx][1]

            # Sample a target object using the concept ranges
            target_object = self.sample_target(concept_ranges)

            # Generate a game instance
            game_instance = self.create_game_instance(target_object, self.all_objects, context_condition)
            game_instances.append(game_instance)

        return game_instances

    def change_one_attribute(input_object, fixed):
        """
        Returns a concept where one attribute is changed.
        Input: A concept consisting of an (example) object and a fixed vector indicating which attributes are fixed in the concept.
        Output: A list of concepts consisting of an (example) object that differs in one attribute from the input object and a new fixed vector.
        """
        changed_concepts = []
        # go through target object and fixed
        # O(n_attributes)
        for i, attribute in enumerate(input_object):
            # check whether attribute in target object is fixed
            if fixed[i] == 1:
                # change one attribute to all possible attributes that don't match the target_object
                # O(n_values)
                for poss_attribute in range(self.properties_dim[i]):
                    # new_fixed = fixed.copy() # change proposed by ChatGPT
                    if poss_attribute != attribute:
                        new_fixed = fixed.copy()  # change proposed by ChatGPT
                        new_fixed[i] = 0
                        changed = list(input_object)
                        changed[i] = poss_attribute
                        # the new fixed values specify where the change took place: (1,1,0) means the change took place in 3rd attribute
                        changed_concepts.append((changed, new_fixed))
        return changed_concepts

    def change_n_attributes(input_object, fixed, n_attributes):
        """
        Changes a given number of attributes from a target object
                given a fixed vector (specifiying the attributes that can and should be changed)
                and a target object
                and a number of how many attributes should be changed.
        """
        changed_concepts = list()
        # O(n_attributes),
        while n_attributes > 0:
            # if changed_concepts is empty, I consider the target_object
            if not changed_concepts:
                changed_concepts = [change_one_attribute(input_object, fixed)]
                n_attributes = n_attributes - 1
            # otherwise consider the changed concepts and change them again	 until n_attributes = 0
            else:
                old_changed_concepts = changed_concepts.copy()
                # O(game_size)
                for sublist in changed_concepts:
                    for changed_concept, fixed in sublist:
                        new_changed_concepts = change_one_attribute(
                            changed_concept, fixed
                        )
                        if new_changed_concepts not in old_changed_concepts:
                            old_changed_concepts.append(new_changed_concepts)
                # copy and store for next iteration
                changed_concepts = old_changed_concepts.copy()
                n_attributes = n_attributes - 1
        # flatten list
        changed_concepts_flattened = [
            changed_concept
            for sublist in changed_concepts
            for changed_concept in sublist
        ]
        # remove doubles
        changed_concepts_final = []
        [
            changed_concepts_final.append(x)
            for x in changed_concepts_flattened
            if x not in changed_concepts_final
        ]
        return changed_concepts_final

    # distractors: number and position of fixed attributes match target concept
    # the more fixed attributes are shared, the finer the context
    distractor_concepts = change_n_attributes(target_objects[0], fixed, sum(fixed))
    # the fixed vectors in the distractor_concepts indicate the number of shared features: (1,0,0) means only first attribute is shared
    # thus sum(fixed) indicates the context condition: from 0 = coarse to n_attributes = fine
    # for the dataset I need objects instead of concepts
    distractor_objects = []
    for dist_concept in distractor_concepts:
        # same fixed vector as for the target concept
        distractor_objects.extend(
            [
                (
                    self.get_all_objects_for_a_concept(
                        self.properties_dim, dist_concept[0], fixed
                    ),
                    tuple(dist_concept[1]),
                )
            ]
        )

    return distractor_objects
