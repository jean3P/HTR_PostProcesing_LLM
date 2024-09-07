import os

# Get the absolute path of the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# HDf5 files

outputs_hdf5 = os.path.join(current_dir, '../../Datasets/outputs')
model_outputs = os.path.join(current_dir, '../models')
results_outputs = os.path.join(current_dir, '../results')
datasets = ['washington', 'bentham', 'iam']