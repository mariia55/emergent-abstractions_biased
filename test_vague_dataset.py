import unittest
import torch
from vague_dataset import VagueDataset  

class TestVagueDataset(unittest.TestCase):

    def setUp(self):
        """Set up a VagueDataset instance before each test."""
        self.dataset = VagueDataset(properties_dim=[3, 3, 3], game_size=10)
        self.sample_input_list = [0.2, 0.5, 0.3]  # Example input list for testing

    def test_normalized_encoding(self):
        """Test if _normalized_encoding correctly normalizes the input list to a tensor whose sum is 1."""
        encoded = self.dataset._normalized_encoding(self.sample_input_list)
        self.assertIsInstance(encoded, torch.Tensor)
        self.assertAlmostEqual(encoded.sum().item(), 1.0)

    def test_get_item(self):
        """Test if get_item correctly retrieves an item (tuple of sender_input, labels, receiver_input) based on a concept index and context condition."""
        item = self.dataset.get_item(0, 0, self.dataset._normalized_encoding)
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 3)

    def test_sample_attribute(self):
        """Test if sample_attribute correctly samples a value within a specified range."""
        attribute = self.dataset.sample_attribute(0.1, 0.9)
        self.assertTrue(0.1 <= attribute <= 0.9)

    def test_sample_target(self):
        """Test if sample_target correctly samples a target object based on given attribute ranges."""
        concept_ranges = [(0.1, 0.5), (0.2, 0.6), (0.3, 0.7)]
        target = self.dataset.sample_target(concept_ranges)
        self.assertTrue(all(0.1 <= attr <= 0.7 for attr in target))

    def test_sample_distractor(self):
        """Test if sample_distractor generates a distractor object based on concept ranges and context condition."""
        concept_ranges = [(0.1, 0.5), (0.2, 0.6), (0.3, 0.7)]
        distractor = self.dataset.sample_distractor(concept_ranges, 'fine')
        self.assertIsInstance(distractor, list)

    def test_get_sample(self):
        """Test if get_sample correctly returns a target object and a list of distractor objects for a given concept index and context condition."""
        target, distractors = self.dataset.get_sample(0, 0)
        self.assertIsInstance(target, list)
        self.assertIsInstance(distractors, list)

    def test_get_all_concepts(self):
        """Test if get_all_concepts successfully retrieves all possible concepts with continuous attributes."""
        concepts = self.dataset.get_all_concepts()
        self.assertIsInstance(concepts, list)

    def test_satisfies(self):
        """Test if satisfies correctly determines whether an object fits within given concept ranges."""
        self.assertTrue(self.dataset.satisfies([0.2, 0.4], [(0.1, 0.3), (0.3, 0.5)]))

    def test_get_all_objects_for_a_concept(self):
        """Test if get_all_objects_for_a_concept generates all possible objects for a given concept based on attribute ranges."""
        feature_ranges = [(0, 1), (0, 1), (0, 1)]
        objects = self.dataset.get_all_objects_for_a_concept([3, 3, 3], feature_ranges)
        self.assertIsInstance(objects, list)

    def test_get_all_possible_objects(self):
        """Test if get_all_possible_objects correctly generates all possible combinations of attribute-feature values."""
        objects = self.dataset.get_all_possible_objects([3, 3, 3])
        self.assertIsInstance(objects, list)

if __name__ == '__main__':
    unittest.main()