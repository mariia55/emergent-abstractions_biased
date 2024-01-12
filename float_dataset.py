from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F
import itertools
import random
from tqdm import tqdm

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)


def generate_all_non_zero_binary_vectors(length: int) -> List[Tuple[int, ...]]:
    """
    Generates all possible binary vectors of a given length, excluding the all-zero vector.

    :param length: The length of the binary vectors.
    :return: A list of tuples, each representing a binary vector.
    """
    # Generate all combinations of 0s and 1s for vectors of the given length
    all_vectors = list(itertools.product([0, 1], repeat=length))

    # Remove the all-zero vector
    all_vectors.remove((0,) * length)

    return all_vectors


@dataclass(frozen=True)
class Concept:
    """Representation of a Concept."""

    concept_object: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    fixed_tuple: Tuple[int, int, int]

    def fits_other_concept_object(
        self, other_concept_object: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> bool:
        """Compare a concepts concept_object against another concept_object."""  # todo: what does this comparison actually mean?
        if 1 not in self.fixed_tuple:
            return False

        same_counter = 0
        required_matches = sum(self.fixed_tuple)

        for fixed_tuple_intger, concept_tensor, other_concept_tensor in zip(
            self.fixed_tuple, self.concept_object, other_concept_object
        ):
            if fixed_tuple_intger == 1:
                if concept_tensor == other_concept_tensor:
                    return False
                same_counter += 1

        return same_counter == required_matches


class FloatDataSet(torch.utils.data.Dataset):
    """
    This class provides the torch.Dataloader-loadable dataset.
    """

    def __init__(
        self,
        properties_dim=None,
        game_size=10,
        device=None,
        testing=False,
        zero_shot=False,
        zero_shot_test=None,
    ):
        """
        properties_dim: vector that defines how many attributes and features per attributes the dataset should contain, defaults to a 3x3x3 dataset
        game_size: integer that defines how many targets and distractors a game consists of
        """

        # Set default values if None was passed
        if properties_dim is None:
            properties_dim = [3, 3, 3]
        if device is None:
            device = "cuda"
        if zero_shot_test is None:
            zero_shot_test = "generic"

        super().__init__()

        self.properties_dim = properties_dim
        self.game_size = game_size
        self.device = device

        # get all concepts
        self.concepts = self.get_all_concepts()
        # get all objects
        self.all_objects = self._get_all_possible_objects()

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
                            self._float_range_encoding,
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
                            self._float_range_encoding,
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
                                    self._float_range_encoding,
                                    include_concept,
                                )
                            )
                        else:
                            train_and_val.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._float_range_encoding,
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
                                    self._float_range_encoding,
                                    include_concept,
                                )
                            )
                        else:
                            train_and_val.append(
                                self.get_item(
                                    concept_idx,
                                    context_condition,
                                    self._float_range_encoding,
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

    def get_sample(self, concept_idx, context_condition):
        """
        Returns a full sample consisting of a set of target objects (target concept)
        and a set of distractor objects (context) for a given concept condition.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        # sample target objects for given game size (if possible, get unique choices)
        try:
            target_objects = random.sample(all_target_objects, self.game_size)
        except ValueError:
            target_objects = random.choices(all_target_objects, k=self.game_size)
        # get all possible distractors for a given concept (for all possible context conditions)
        context = self.get_distractors(concept_idx, context_condition)
        context_sampled = self.sample_distractors(context, context_condition)
        # return target concept, context (distractor objects + context_condition) for each context
        return [target_objects, fixed], context_sampled

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
        poss_dist = self.all_objects

        for obj in poss_dist:
            # find out how many attributes are shared between the possible distractor object and the target concept
            # (by only comparing fixed attributes because only these are relevant for defining the context)
            shared = sum(
                1
                for idx in fixed_attr_indices
                if obj[idx] == all_target_objects[0][idx]
            )
            if shared == context_condition:
                context.append(obj)

        return context

    def sample_distractors(self, context, context_condition):
        """
        Function for sampling the distractors from a specified context condition.
        """
        # sample distractor objects for given game size and the specified context condition
        # distractors = [dist_obj for dist_objs in context for dist_obj in dist_objs]
        context_new = []
        try:
            context_new.append(
                [random.sample(context, self.game_size), context_condition]
            )
        except ValueError:
            context_new.append(
                [random.choices(context, k=self.game_size), context_condition]
            )
        return context_new

    def sample_distractors_old(self, distractors, fixed):
        """
        Function for sampling the distractors from all possible context conditions.
        """
        # sample distractor objects for given game size and each context condition (constrained by level of abstraction)
        context = list()
        context_candidates = list()
        for i in range(sum(fixed)):
            for dist_objects, context_condition in distractors:
                # check for context condition
                # sum(context_condition) gives the number of shared attributes
                if sum(context_condition) == i:
                    for dist_object in dist_objects:
                        context_candidates.append([dist_object, i])
        helper_i = 0
        helper_list = list()
        # for i in range(len(self.properties_dim)):
        for i, (dist_object, context_condition) in enumerate(context_candidates):
            if helper_i == context_condition:
                # gather all objects belonging to the same context condition
                helper_list.append(dist_object)
                # final index: should be sampled
                if i == len(context_candidates) - 1:
                    try:
                        context.append(
                            [random.sample(helper_list, self.game_size), helper_i]
                        )
                    except ValueError:
                        context.append(
                            [random.choices(helper_list, k=self.game_size), helper_i]
                        )
            # catch the final context condition as well
            elif context_condition == len(self.properties_dim) - 1:
                try:
                    context.append(
                        [random.sample(helper_list, self.game_size), helper_i]
                    )
                except ValueError:
                    context.append(
                        [random.choices(helper_list, k=self.game_size), helper_i]
                    )
                helper_i = helper_i + 1
                helper_list = list()
                helper_list.append(dist_object)
            # when moving to the next context condition, first sample from the old
            else:
                # sample from all objects belonging to the same context condition
                try:
                    context.append(
                        [random.sample(helper_list, self.game_size), helper_i]
                    )
                except ValueError:
                    context.append(
                        [random.choices(helper_list, k=self.game_size), helper_i]
                    )
                helper_i = helper_i + 1
                helper_list = list()
                helper_list.append(dist_object)
        return context

    def get_all_concepts(self):
        """
        Returns all possible concepts for a given dataset size.
        Concepts consist of (objects, fixed) tuples
                objects: a list with all object-tuples that satisfy the concept
                fixed: a tuple that denotes how many and which attributes are fixed
        """
        fixed_vectors = self.get_fixed_vectors()
        all_objects = self._get_all_possible_objects()
        # create all possible concepts
        all_fixed_object_pairs = list(itertools.product(all_objects, fixed_vectors))

        concepts_list = list()
        # go through all concepts (i.e. fixed, objects pairs)
        for object, fixed_vector in all_fixed_object_pairs:
            # treat each fixed_object pair as a target concept once
            # e.g. target concept (_, _, 0) (i.e. fixed = (0,0,1) and objects e.g. (0,0,0), (1,0,0))
            # go through all objects and check whether they satisfy the target concept (in this example have 0 as 3rd attribute)
            target_objects = set()
            for obj in all_objects:
                concept = Concept(concept_object=object, fixed_tuple=fixed_vector)
                if concept.fits_other_concept_object(other_concept_object=obj):
                    target_objects.add(obj)
            # concepts are tuples of fixed attributes and all target objects that satisfy the concept
            target_objects = list(target_objects)
            if (target_objects, fixed_vector) not in concepts_list:
                concepts_list.append((target_objects, fixed_vector))
        return concepts_list

    def get_shared_vectors(self, fixed):
        """
        Returns fixed vectors for all possible context conditions based on a concept (i.e. the fixed vector).
        These are called "shared_vectors" because the number and position of attributes which are shared with the
        target concept define the context condition. The more fixed attributes are shared, the finer the context.
        """
        shared_vectors = []
        for i, attribute in enumerate(fixed):
            shared = list(itertools.repeat(0.0, len(fixed)))
            if attribute == 1:
                shared[i] = 1.0
                shared_vectors.append(shared)
        return shared_vectors

    def get_fixed_vectors(self):
        """
        Returns all possible fixed vectors for a given dataset size.
        Fixed vectors are vectors of length len(properties_dim), where 1 denotes that an attribute is fixed, 0 that it isn't.
        The more attributes are fixed, the more specific the concept -- the less attributes fixed, the more generic the concept.
        """
        # what I want to get: [(1,0,0), (0,1,0), (0,0,1)] for most generic
        # concrete: [(1,1,0), (0,1,1), (1,0,1)]
        # most concrete: [(1,1,1)]
        # for variable dataset sizes
        vector_length = len(self.properties_dim)
        return generate_all_non_zero_binary_vectors(length=vector_length)

    @staticmethod
    def get_all_objects_for_a_concept(properties_dim, features, fixed):
        """
        Returns all possible objects for a concept at a given level of abstraction
        features: Defines the features which are fixed
        fixed: Defines how many and which attributes are fixed
        """
        # Generate all possible combinations of attribute-feature values as float lists
        list_of_dim = [torch.linspace(0.1, 0.9, dim) for dim in properties_dim]
        all_objects = list(itertools.product(*list_of_dim))

        # Filter objects based on fixed attributes and features
        concept_objects = []

        # Account for the case where 0 attributes should be shared in context_condition 0
        if not 1 in fixed:
            return all_objects

        # Determine the indices of attributes that should be fixed
        fixed_indices = [i for i, is_fixed in enumerate(fixed) if is_fixed == 1]

        # Filter objects that match the fixed features for each fixed attribute
        for index in fixed_indices:
            concept_objects.extend(
                [obj for obj in all_objects if obj[index] == features[index]]
            )

        return concept_objects

    def _get_all_possible_objects(self):
        """
        Returns all possible combinations of attribute-feature values as a list of lists.
        """
        list_of_dim = [
            range(0, dim) for dim in self.properties_dim
        ]  # Generate float values between 0 and 1
        # Generate all possible combinations of floats for each dimension
        all_objects = list(
            itertools.product(*list_of_dim)
        )  # todo: here, the all-zeros vector is not excluded, why? (in get_fixed_vectors it is excluded)
        return all_objects

    def _many_hot_encoding(self, input_list):
        """
        Outputs a binary one-dim vector.
        """
        output = torch.zeros([sum(self.properties_dim)]).to(device=self.device)
        start = 0

        for elem, dim in zip(input_list, self.properties_dim):
            # Scale the float value to the range [0, dim]
            scaled_value = int(elem * dim)
            # Ensure the scaled value is within the valid range
            scaled_value = max(0, min(dim - 1, scaled_value))
            output[start + scaled_value] = 1
            start += dim

        return output

    def _float_range_encoding(self, input_list, float_range=(0.1, 0.9)):
        """
        Outputs a vector where each attribute is represented by a random float from a specified range.
        """
        output = torch.zeros([sum(self.properties_dim)]).to(device=self.device)
        start = 0
        float_range_difference = float_range[1] - float_range[0]

        for elem, dim in zip(input_list, self.properties_dim):
            # Determines the range segment for the current attribute
            segment_range = (float_range_difference) / dim
            # Calculates the specific range for the current value
            value_range_start = float_range[0] + elem * segment_range
            value_range_end = value_range_start + segment_range

            selected_float = random.uniform(value_range_start, value_range_end)

            # Places the selected float in the corresponding position in the output vector
            output[start : start + dim] = torch.tensor(
                [selected_float if i == int(elem) else 0 for i in range(dim)]
            )
            start += dim

        return output

    def get_distractors_old(self, concept_idx):
        """
        Returns all possible distractor objects for each context based on a given target concept.
        return (context, distractor_objects) tuples
        """

        target_objects, fixed = self.concepts[concept_idx]
        fixed = list(fixed)

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
