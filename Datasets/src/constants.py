import os

# Get the absolute path of the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the resources directory relative to the current script
resources_path = os.path.join(current_dir, '../resources')


# Define the absolute paths for the datasets
bentham_path = os.path.join(resources_path, 'BenthamDatasetR0-GT')
iam_path = os.path.join(resources_path, 'IAM')
washington_path = os.path.join(resources_path, 'washington')

outputs_path = os.path.join(current_dir, '../outputs')
splits_bentham_path = os.path.join(outputs_path, 'bentham')
splits_washington_path = os.path.join(outputs_path, 'washington')
splits_iam_path = os.path.join(outputs_path, 'iam')

