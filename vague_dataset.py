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
        Creates the train, validation, and test datasets based on the number of possible concepts.
        """
        if sum(split_ratio) != 1:
            raise ValueError("Split ratio must sum up to 1.")

        train_ratio, val_ratio, _ = split_ratio

        # Shuffle concept indices
        concept_indices = torch.randperm(len(self.concepts)).tolist()
        # Split based on how many distinct concepts there are
        ratio = int(len(self.concepts) * (train_ratio + val_ratio))

        train_and_val = []
        print("Creating train_ds and val_ds...")
        for concept_idx in tqdm(concept_indices[:ratio]):
            for _ in range(self.game_size):
                # Consider all possible context conditions for each concept
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                for context_condition in range(nr_possible_contexts):
                    item = self.get_item(concept_idx, context_condition, self._normalized_encoding, include_concept)
                    train_and_val.append(item)

        # Calculate the number of samples for training and validation
        train_samples = int(len(train_and_val) * train_ratio / (train_ratio + val_ratio))
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])

        # Information about the train dataset
        train.dimensions = self.properties_dim

        test = []
        print("\nCreating test_ds...")
        for concept_idx in tqdm(concept_indices[ratio:]):
            for _ in range(self.game_size):
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                for context_condition in range(nr_possible_contexts):
                    item = self.get_item(concept_idx, context_condition, self._normalized_encoding, include_concept)
                    test.append(item)

        return train, val, test

    def get_zero_shot_datasets(self, split_ratio, test_cond='generic', include_concept=False):
        """
        Note: Generates train, val, and test data.
            Test and training set contain different concepts. There are two possible datasets:
            1) 'generic': train on more specific concepts, test on most generic concepts
            2) 'specific': train on more generic concepts, test on most specific concepts
        :param split_ratio Tuple of ratios (train, val) of the samples should be in the training and validation sets.
        """

        if sum(split_ratio) != 1:
            raise ValueError("Split ratio must sum up to 1.")

        train_ratio, val_ratio = split_ratio

        train_and_val = []
        test = []

        print("Creating train_ds, val_ds and test_ds...")
        for concept_idx in tqdm(range(len(self.concepts))):
            for _ in range(self.game_size):
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                for context_condition in range(nr_possible_contexts):
                    # For 'generic' test condition
                    if test_cond == 'generic':
                        if nr_possible_contexts == 1:  # Most generic concepts
                            item = self.get_item(concept_idx, context_condition, self._normalized_encoding, include_concept)
                            test.append(item)
                        else:
                            item = self.get_item(concept_idx, context_condition, self._normalized_encoding, include_concept)
                            train_and_val.append(item)

                    # For 'specific' test condition
                    if test_cond == 'specific':
                        if nr_possible_contexts == len(self.properties_dim):  # Most specific concepts
                            item = self.get_item(concept_idx, context_condition, self._normalized_encoding, include_concept)
                            test.append(item)
                        else:
                            item = self.get_item(concept_idx, context_condition, self._normalized_encoding, include_concept)
                            train_and_val.append(item)

        # Splitting train and validation datasets
        train_samples = int(len(train_and_val) * train_ratio)
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])

        # Information about the train dataset
        train.dimensions = self.properties_dim
        print("Length of train and validation datasets:", len(train), "/", len(val))
        print("Length of test dataset:", len(test))

        return train, val, test

    def get_item(self, concept_idx, context_condition, encoding_func, include_concept=False):
        """
        Receives concept-context pairs and an encoding function.
        Returns encoded (sender_input, labels, receiver_input).
            sender_input: (sender_input_objects, sender_labels)
            labels: indices of target objects in the receiver_input
            receiver_input: receiver_input_objects
        The sender_input_objects and the receiver_input_objects are different objects sampled from the same concept 
        and context condition.
        """
        # Use get_sample() to get sampled target and distractor objects 
        # The concrete sampled objects can differ between sender and receiver.
        sender_target, sender_distractors = self.get_sample(concept_idx, context_condition)
        receiver_target, receiver_distractors = self.get_sample(concept_idx, context_condition)

        # Initialize sender and receiver input with target objects only
        sender_input = [sender_target]
        receiver_input = [receiver_target]

        # Append context objects (distractors) based on the context condition
        sender_input.extend(sender_distractors)
        receiver_input.extend(receiver_distractors)

        # Shuffle receiver input and create labels
        random.shuffle(receiver_input)
        receiver_label = [idx for idx, obj in enumerate(receiver_input) if obj == receiver_target]
        # note alternative: receiver_label = [1 if obj == receiver_target else 0 for obj in receiver_input] 

        # Convert receiver labels to tensor and apply one-hot encoding
        receiver_label = torch.Tensor(receiver_label).to(torch.int64).to(device=self.device)
        receiver_label = F.one_hot(receiver_label, num_classes=self.game_size*2).sum(dim=0).float()

        # Encode and return as tensor
        sender_input = torch.stack([encoding_func(obj) for obj in sender_input])
        receiver_input = torch.stack([encoding_func(obj) for obj in receiver_input])

        # Output needs to have the structure sender_input, labels, receiver_input
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
    