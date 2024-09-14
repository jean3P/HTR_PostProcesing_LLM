# constants.py

import os

# Get the absolute path of the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

Datasets_path = os.path.join(current_dir, '../../Datasets/src')
TrOcr_path = os.path.join(current_dir, '../../TrOCR_model/src')
LLMs_path = os.path.join(current_dir, '../../LLMs/src')
