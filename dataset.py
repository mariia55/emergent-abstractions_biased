# code inspired by https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/dataset.py

import torch
import pandas as pd
import itertools


class DataSet(torch.utils.data.Dataset): # question: Is there a reason not to use the torch dataset?
	""" 
	This class provides the torch.Dataloader-loadable dataset.
	"""
	def __init__(self, properties_dim=[3,3,3], game_size=3):
		"""
		properties_dim: vector that defines how many attributes and features per attributes the dataset should contain, defaults to a 3x3x3 dataset
		game_size: integer that defines how many targets and distractors a game consists of
		"""
		super().__init__()
		
		self.properties_dim = properties_dim
		
		# get all possible concept-context pairs (depends on dataset size) - full sets of objects, later sample from these
		self.concept_context_pairs = self.get_concept_context_pairs(self)
		
		
		# define pairs of targets and distractors
		
		# retrieve all possible objects
		#self.objects = self._get_all_possible_objects(properties_dim)
		#print(self.objects)
		
		# create all possible games (i.e., target-distractor-combinations)
		# define concepts
		#self.concepts = self.define_targets(properties_dim)
		# define contexts
		#self.contexts = self.define_distractors(self.properties_dim, self.objects, self.concepts)
		
		
		fixed = [0,0,1]
		features = [2,0,1]
		objects_for_a_concept = self.get_all_objects_for_a_concept(self.properties_dim, fixed, features)
		#print(objects_for_a_concept)
	
		
	@staticmethod
	def get_concept_context_pairs(self):
		"""
		Returns all possible concept-context pairs for a given dataset size in the form of fixed and features vectors.
		Input: properties_dim
		Output: 
		Concepts (contexts are also treated as concepts) consist of fixed-features tuples: 
			fixed: vector of length len(properties_dim), 1 denotes that the attribute is fixed, 0 denotes unfixed attribute
			features: vector of length len(properties_dim) that fixes the attributes to specific feature values
		"""		
		fixed_vectors = self.get_fixed_vectors(self.properties_dim)		
		print(fixed_vectors)
		# features are just all possible objects
		features_vectors = self._get_all_possible_objects(self.properties_dim)
		#print(features_vectors)
		# match fixed and features vectors (and sort according to level of abstraction?)
		concepts = list(itertools.product(fixed_vectors, features_vectors))
		#print(concepts)
		# distractor concepts: number and position of fixed attributes match target concept
		# the more fixed attributes are shared, the finer the context
		
		#print(sum(fixed_vectors[2])) # easy way to check the level of abstraction (1 is most generic, n is most specific)
		# maybe first build concept-context pairs and then match fixed and feature vectors 
		generic_concepts = list(itertools.product(fixed_vector for fixed_vector in fixed_vectors if sum(fixed_vector) == 1))
		print(generic_concepts) 
		
		
	@staticmethod
	def get_fixed_vectors(properties_dim):
		"""
		Returns all possible fixed vectors for a given dataset size.
		Fixed vectors are vectors of length len(properties_dim), where 1 denotes that an attribute is fixed, 0 that it isn't.
		The more attributes are fixed, the more specific the concept -- the less attributes fixed, the more generic the concept.
		"""
		# what I want to get: [(1,0,0), (0,1,0), (0,0,1)] for most generic
		# concrete: [(1,1,0), (0,1,1), (1,0,1)]
		# most concrete: [(1,1,1)]
		# for variable dataset sizes
		
		# range(0,2) because I want [0,1] values for whether an attribute is fixed or not		
		list_of_dim = [range(0, 2) for dim in properties_dim]
		fixed_vectors = list(itertools.product(*list_of_dim))
		# remove first element (0,..,0) as one attribute always has to be fixed
		fixed_vectors.pop(0)
		return fixed_vectors
				
		
	@staticmethod
	def get_all_objects_for_a_concept(properties_dim, fixed, features):
		"""
		Returns all possible objects for a concept at a given level of abstraction
		fixed: Defines how many and which attributes are fixed
		features: Defines the features which are fixed
		"""
		# retrieve all possible objects
		list_of_dim = [range(0, dim) for dim in properties_dim]
		all_objects = list(itertools.product(*list_of_dim))
		
		# get concept objects
		concept_objects = list()
		
		# determine the indices of attributes that should be fixed
		fixed_indices = list(itertools.compress(range(0,len(fixed)), fixed))
		# find possible concepts for each index
		possible_concepts = dict()
		for index in fixed_indices:
			possible_concepts[index] = ([object for object in all_objects if object[index] == features[index]])
		#print(possible_concepts)	
	
		# keep only those that also match with the other fixed features, i.e. that are possible concepts for all fixed indices
		all = list(possible_concepts.values())
		concept_objects = list(set(all[0]).intersection(*all[1:]))
		
		return concept_objects
		
		
	
	@staticmethod
	def _get_all_possible_objects(properties_dim):
		"""
		Returns all possible combinations of attribute-feature values as a dataframe.
		"""
		list_of_dim = [range(0, dim) for dim in properties_dim]
		# Each object is a row
		all_objects = list(itertools.product(*list_of_dim))
		return all_objects#pd.DataFrame(all_objects)
		
		
	@staticmethod
	def define_targets(properties_dim):
		"""
		Defines all possible target concepts on different levels of specificity according to the properties dimension vector.
		"""
				
		
		
		
	def define_distractors(properties_dim, objects, concepts):
		"""
		Defines the distractor concepts (context) for different context granularities (fine -> coarse).
		"""
		# distractor concepts fix the same attributes as target concepts but with a different feature value
		# the more attributes
		
		
	
	def get_item(self, object_idx):
		"""
		Overwrite get_item().
		"""
		return null
     
