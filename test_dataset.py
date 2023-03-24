import unittest
import math
import numpy as np

from dataset import DataSet


class TestDataset(unittest.TestCase):

    def setUp(self):

        self.possible_properties = [[2, 2], [4, 4],
                                    [2, 2, 2], [4, 4, 4],
                                    [2, 2, 2, 2], [4, 4, 4, 4]]
        self.game_sizes = [1, 3, 5, 10]

        self.datasets = []
        for props in self.possible_properties:
            for size in self.game_sizes:
                self.datasets.append(DataSet(props, size))

    def test_get_all_concepts(self):
        """
        Test
        - if number of concepts is correct
        - if number of instances per concept is correct

        Functions used by get_all_concepts:
        - get_fixed_vectors
        - _get_all_possible_objects
        - get_all_objects_for_a_concepts
        - satisfies
        """

        for ds in self.datasets:
            n_atts = len(ds.properties_dim)
            n_vals = ds.properties_dim[0]
            concepts = ds.concepts

            # Test total number of concepts
            combinations_per_attribute = 0
            for i in range(1, n_atts+1):
                combinations_per_attribute += math.comb(n_atts, i)*(n_vals**i)
            total_combinations = combinations_per_attribute
            print(len(ds.concepts), total_combinations)
            self.assertEqual(total_combinations, len(concepts))

            # Test number of instances per concept
            for c in concepts:
                n_fixed = np.sum(c[1])
                n_instances = n_vals**(n_atts-n_fixed)
                self.assertEqual(n_instances, len(c[0]))

    def test_get_distractors(self):
        """
        Test
        - if the right number (all) distractors are generated for each concept
        - if at least one fixed attribute is different between target and distractor
        """

        for ds in self.datasets:
            n_atts = len(ds.properties_dim)
            n_vals = ds.properties_dim[0]
            concepts = ds.concepts
            n_objects = n_vals**n_atts

            for i, concept in enumerate(concepts):
                distractors_distributed = DataSet.get_distractors(ds, i)
                distractors = []
                for elem in distractors_distributed:
                    distractors += elem[0]

                # assert number of distractors correct
                n_fixed = np.sum(concept[1])
                n_expected_distractors = int(n_objects * (1 - (1 / n_vals) ** n_fixed))
                self.assertEqual(n_expected_distractors, len(distractors))

                # assert distractors do not correspond to the target
                # self.assertFalse(concept[0] in distractors)

                # assert that distractors differ from target in at least one fixed attribute
                for d in distractors:
                    diff = np.abs(np.array(concept[0]) - np.array(d))
                    # select the difference between target and distractor for fixed attributes
                    mask = 1 - np.tile(np.array(concept[1]), (len(diff), 1))
                    masked_difference = np.ma.masked_array(diff, mask)
                    self.assertTrue(0 not in np.sum(masked_difference, axis=1))

    def test_get_item(self):
        pass

    def test_get_dataset(self):
        pass


if __name__ == '__main__':

    unittest.main()