from dataset import DataSet

dataset = DataSet(properties_dim=[4, 4, 4]) 
unique_concepts = dataset.get_all_concepts()
number_of_unique_concepts = len(unique_concepts)
print("Number of unique concepts:", number_of_unique_concepts)
