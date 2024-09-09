# src/constants.py

import os

# Get the absolute path of the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

results_from_model_path = os.path.join(current_dir, '../../TrOCR_model/results')
results_llm = os.path.join(current_dir, '../results')
training_suggestion_path = os.path.join(current_dir, '../training_sets_files')