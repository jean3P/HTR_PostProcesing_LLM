import h5py

file_path = '../models/washington/train_25/model_flor/checkpoint_weights.hdf5'

with h5py.File(file_path, 'r') as hdf:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Data type: {obj.dtype}")

    hdf.visititems(print_structure)
