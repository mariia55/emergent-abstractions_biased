import shutil
import os

def ignore_pkl_files(src, names):
    """
    Ignore .pkl files in the source directory.
    """
    ignored_names = [name for name in names if name.endswith('.pkl')]
    return ignored_names

gran = 'fine'
data = '(5,4)'

# Define the destination folder and ensure it exists
destination_folder = os.path.join(
    r'C:\Users\Colab\Desktop\interactions', 
    data, 
    'granularity_' + gran
)
os.makedirs(destination_folder)  # Creates the directory and its parents if they don't exist

# Define the source folder
source_folder = os.path.join(
    'results', 
    data + '_game_size_10_vsf_3', 
    'standard', 
    'granularity_' + gran
)

# Copy the entire directory tree, excluding .pkl files
try:
    shutil.copytree(source_folder, destination_folder, ignore=ignore_pkl_files, dirs_exist_ok=True)
    print(f"Directory copied from {source_folder} to {destination_folder}, excluding .pkl files.")
except FileExistsError:
    print(f"Destination directory {destination_folder} already exists.")
except FileNotFoundError:
    print(f"Source directory {source_folder} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
