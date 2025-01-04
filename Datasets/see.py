import h5py

file_path = 'outputs/washington/combined_washington_dataset_htr_old.hdf5'

with h5py.File(file_path, 'r') as hdf:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Data type: {obj.dtype}")


    hdf.visititems(print_structure)


import h5py
import numpy as np

file_path = 'outputs/washington/combined_washington_dataset_htr_old.hdf5'

with h5py.File(file_path, 'r') as hdf:
    # Choose the partition you want to sample from, e.g., 'test', 'train_25', etc.
    partition = 'test'

    # Access the datasets
    dt_dataset = hdf[f"{partition}/dt"]
    gt_dataset = hdf[f"{partition}/gt"]
    path_dataset = hdf[f"{partition}/path"]

    # Get a small sample of the first 3 entries
    dt_sample = dt_dataset[:3]   # This will be numpy arrays (images)
    gt_sample = gt_dataset[:3]   # Ground truth strings
    path_sample = path_dataset[:3]  # Image file names or paths

    # Print or inspect the samples
    print("DT Sample Shapes and Types:")
    for i, img in enumerate(dt_sample):
        print(f"Index {i}: img shape={img.shape}, dtype={img.dtype}")

    print("\nGT Sample:")
    for i, gt in enumerate(gt_sample):
        print(f"Index {i}: gt={gt}")

    print("\nPath Sample:")
    for i, pth in enumerate(path_sample):
        print(f"Index {i}: path={pth}")
