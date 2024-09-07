import graphene
import os.path
from constants import splits_bentham_path, splits_washington_path, splits_iam_path
from h5py import File
import logging  # Add logging to track the path_data

# Configure logging
logging.basicConfig(level=logging.INFO)


# Define the FileInfo type that corresponds to the data in the HDF5 file
class FileInfo(graphene.ObjectType):
    file_name = graphene.String()
    ground_truth = graphene.String()
    image_data = graphene.List(graphene.Float)  # Assuming the image data can be represented as floats


# Define the PartitionData type to hold the data and additional metadata like the total count
class PartitionData(graphene.ObjectType):
    total_count = graphene.Int()  # Total number of elements in the partition
    global_total = graphene.Int()  # Total number of elements across all partitions
    data = graphene.List(FileInfo)  # The actual partition data
    path = graphene.String()  # The full path of the image


# Define the Query class for fetching HDF5 data
class Query(graphene.ObjectType):
    partition_data = graphene.Field(
        PartitionData,
        partition=graphene.String(required=True),  # Partition can be 'train_100', 'train_75', 'train_50', 'train_25', 'valid', or 'test'
        name_dataset=graphene.String(required=True),  # Name of the dataset (e.g., 'bentham')
        number_of_rows=graphene.Int(default_value=10)  # Limit the number of rows returned
    )

    # Resolver to get the data for a specific partition and dataset
    def resolve_partition_data(self, info, partition, name_dataset, number_of_rows):
        if name_dataset == 'bentham':
            hdf5_path = os.path.join(splits_bentham_path, f'{name_dataset}_dataset.hdf5')
        elif name_dataset == 'washington':
            hdf5_path = os.path.join(splits_washington_path, f'{name_dataset}_dataset.hdf5')
        elif name_dataset == 'iam':
            hdf5_path = os.path.join(splits_iam_path, f'{name_dataset}_dataset.hdf5')
        else:
            raise ValueError(f"Dataset '{name_dataset}' not supported.")

        # Ensure the HDF5 file exists
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"The dataset file {hdf5_path} does not exist.")

        # Open and read the HDF5 file
        with File(hdf5_path, "r") as f:
            # Check if the partition exists in the file
            if partition not in f:
                raise ValueError(f"Partition '{partition}' not found in the dataset.")

            # Get the global total count across all partitions (train subsets + valid + test)
            global_total = sum(
                len(f[f"{pt}/dt"]) for pt in ['train_100', 'train_75', 'train_50', 'train_25', 'valid', 'test'])

            # Get the total count of elements in the specified partition
            total_count = len(f[f"{partition}/dt"])

            # Extract the full image path (stored as an HDF5 attribute)
            full_path = f.attrs['full_image_path']  # This is already a string, no need to decode

            # Extract data for the specified partition
            dt_data = f[f"{partition}/dt"][:number_of_rows]  # Limit the data to `number_of_rows`
            gt_data = f[f"{partition}/gt"][:number_of_rows]
            path_data = f[f"{partition}/path"][:number_of_rows]

            # Decode byte strings into UTF-8 strings where necessary
            decoded_path_data = [path.decode('utf-8') if isinstance(path, bytes) else path for path in path_data]
            decoded_gt_data = [gt.decode('utf-8') if isinstance(gt, bytes) else gt for gt in gt_data]

            # Log the content of path_data for debugging
            logging.info(
                f"Contents of path_data for partition '{partition}': {decoded_path_data}")

            # Convert the data and return it along with the total count information
            return PartitionData(
                total_count=total_count,
                global_total=global_total,
                path=full_path,  # Include full path in the response
                data=[
                    FileInfo(
                        file_name=path,  # Path is now a decoded string
                        ground_truth=gt,  # Ground truth is now a decoded string
                        image_data=list(dt.flatten())  # Flatten the image data into a list
                    )
                    for dt, gt, path in zip(dt_data, decoded_gt_data, decoded_path_data)
                ]
            )



# Define the schema
schema = graphene.Schema(query=Query)
