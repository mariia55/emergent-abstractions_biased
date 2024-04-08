import torch
import matplotlib.pyplot as plt


# Specify the path to your .ds file
file_path = "C:\\Users\\annab\\OneDrive\\Documenti\\GitHub\\emergent-abstractions\\data\\dim(3,3)_granularity_mixed.ds"

# Load the dataset from the .ds file
with open(file_path, "rb") as f:
    dataset = torch.load(f)

# Now you can use the 'dataset' object as you normally would
# For example, you can access its attributes and methods
print(dataset.properties_dim)
print(len(dataset))

# You can also iterate through the dataset to access individual samples
for sample in dataset:
    # Process each sample as needed
    print(sample)