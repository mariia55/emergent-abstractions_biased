# code inspired by https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/dataset.py

import torch
import pandas as pd
import itertools
import random

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
		self.game_size = game_size
		
		# get all concepts
		self.concepts = self.get_all_concepts(self)
		#print(self.concepts)
		
		# hierarchical reference game:
		#get_sample(self, sender_object_idx, relevance) returns sender_object=sender_input, target, distractors -> creates distractors based on relevance vectors and sender object and game size!
		#get_item(self, object_idx, relevance, encoding_func) returns (sender_input, relevance), label, receiver_input=distractors+target
		#get_datasets(self, split_ratio) uses get_item to create datasets
		
		# Where do I specify the context condition?
		sample = self.get_sample(self, 20)
		print(sample)
		
		
		
	@staticmethod
	def get_sample(self, concept_idx):
		"""
		Returns a full sample consisting of a set of target objects (target concept) and a set of distractor objects (context) for a given concept.
		"""
		print(self.concepts[concept_idx])
		all_target_objects = self.concepts[concept_idx][0]
		print(all_target_objects)
		# sample target objects for given game size (if possible, get unique choices)
		try:
			target_objects = random.sample(all_target_objects, self.game_size)
		except ValueError:
			target_objects = random.choices(all_target_objects, k=self.game_size)
		print("sampled target objects", target_objects)
		distractors = self.get_distractors(self, concept_idx)
		print("distractors", distractors)
		# sample distractor objects for given game size and given context condition
		
		
		
	@staticmethod
	def get_distractors(self, concept_idx):
		"""
		Returns distractor objects for each context based on a given target concept and game size (i.e. number of targets and distractors).
		return (context, distractor_objects) tuples
		"""
		
		def change_one_attribute(input_object, fixed):
			"""
			Returns a concept where one attribute is changed.
			Input: A concept consisting of an (example) object and a fixed vector indicating which attributes are fixed in the concept. 
			Output: A list of concepts consisting of an (example) object that differs in one attribute from the input object and a new fixed vector.
			"""
			changed_concepts = list()
			# go through target object and fixed
			for i, attribute in enumerate(input_object):
				# check whether attribute in target object is fixed
				if fixed[i] == 1:
					# change one attribute to all possible attributes that don't match the target_object
					for poss_attribute in range(self.properties_dim[i]):
						new_fixed = fixed.copy()
						if poss_attribute != attribute:
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
			while(n_attributes > 0):
				# if changed_concepts is empty, I consider the target_object
				if not changed_concepts:
					changed_concepts = [change_one_attribute(input_object, fixed)]
					n_attributes = n_attributes -1
				# otherwise consider the changed concepts and change them again	 until n_attributes = 0
				else:
					old_changed_concepts = changed_concepts.copy()
					for sublist in changed_concepts:
						for (changed_concept, fixed) in sublist:
							new_changed_concepts = change_one_attribute(changed_concept, fixed)
							if new_changed_concepts not in old_changed_concepts:
								old_changed_concepts.append(new_changed_concepts)
					# copy and store for next iteration
					changed_concepts = old_changed_concepts.copy()
					n_attributes = n_attributes -1
			# flatten list
			changed_concepts_flattened = [changed_concept for sublist in changed_concepts for changed_concept in sublist]
			# remove doubles
			changed_concepts_final = []
			[changed_concepts_final.append(x) for x in changed_concepts_flattened if x not in changed_concepts_final]
			return changed_concepts_final
			
		target_objects, fixed = self.concepts[concept_idx]
		fixed = list(fixed)
		# distractors: number and position of fixed attributes match target concept
		# the more fixed attributes are shared, the finer the context
		distractor_concepts = change_n_attributes(target_objects[0], fixed, sum(fixed))
		# the fixed vectors in the distractor_concepts indicate the number of shared features: (1,0,0) means only first attribute is shared
		# thus sum(fixed) indicates the context condition: from 0 = coarse to n_attributes = fine
		# for the dataset I need objects instead of concepts
		distractor_objects = list()
		for dist_concept in distractor_concepts:
			# same fixed vector as for the target concept
			distractor_objects.extend([self.get_all_objects_for_a_concept(self.properties_dim, dist_concept[0], fixed), dist_concept[1]])
		return distractor_objects
		
		
		
	@staticmethod
	def get_all_concepts(self):
		"""
		Returns all possible concepts for a given dataset size.
		Concepts consist of (objects, fixed) tuples
			objects: a list with all object-tuples that satisfy the concept
			fixed: a tuple that denotes how many and which attributes are fixed
		"""
		fixed_vectors = self.get_fixed_vectors(self.properties_dim)		
		#print(fixed_vectors)
		all_objects = self._get_all_possible_objects(self.properties_dim)
		#print(all_objects)
		# create all possible concepts
		all_fixed_object_pairs = list(itertools.product(all_objects, fixed_vectors))
		#print(all_fixed_object_pairs)
		
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
		return concepts
		
		
	@staticmethod
	def satisfies(object, concept):
		"""
		Checks whether an object satisfies a target concept, returns a boolean value.
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
	def get_all_objects_for_a_concept(properties_dim, features, fixed):
		"""
		Returns all possible objects for a concept at a given level of abstraction
		features: Defines the features which are fixed
		fixed: Defines how many and which attributes are fixed
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
     
