import json
import os
import pandas as pd

from constants import results_llm
from utils.aux_processing import detect_immediate_repeated_words, detect_close_repeated_word_sequences

name_dataset = 'bentham'
base_dir = os.path.join(results_llm, name_dataset, 'Flor_model', 'gpt-4o-mini', 'method_1')

# Initialize a list to collect data for the table
summary_data = []

# Iterate over subdirectories (train_25, train_50, train_75, train_100)
for sub_dir in ['train_25', 'train_50', 'train_75', 'train_100']:
    dir_path = os.path.join(base_dir, sub_dir)
    if os.path.exists(dir_path):
        print(f"\nProcessing {sub_dir} directory:")

        # Initialize counters for good and bad corrections comparing OCR to GT
        total_good_ocr_corrections = 0
        total_bad_ocr_corrections = 0

        # Iterate over all JSON files in the subdirectory
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(dir_path, file_name)

                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                    # Iterate through each entry in the JSON file
                    for entry in data:
                        # Ground Truth Label
                        ground_truth_label = entry.get("ground_truth_label")
                        ground_truth_repeated = detect_immediate_repeated_words(ground_truth_label)
                        ground_truth_close_repeated = detect_close_repeated_word_sequences(ground_truth_label)

                        # OCR Predicted Label
                        ocr_predicted_label = entry["OCR"]["predicted_label"]
                        ocr_repeated = detect_immediate_repeated_words(ocr_predicted_label)
                        ocr_close_repeated = detect_close_repeated_word_sequences(ocr_predicted_label)

                        # Compare OCR directly to the Ground Truth

                        # Good OCR prediction: OCR matches the GT for duplicated words
                        if (ground_truth_repeated or ground_truth_close_repeated) == (ocr_repeated or ocr_close_repeated):
                            total_good_ocr_corrections += 1
                        else:
                            # Bad OCR prediction: OCR output does not match the GT for duplicates
                            total_bad_ocr_corrections += 1

        # Calculate proportions
        total_ocr_comparisons = total_good_ocr_corrections + total_bad_ocr_corrections
        if total_ocr_comparisons > 0:
            good_ocr_proportion = (total_good_ocr_corrections / total_ocr_comparisons) * 100
            bad_ocr_proportion = (total_bad_ocr_corrections / total_ocr_comparisons) * 100
        else:
            good_ocr_proportion = 0
            bad_ocr_proportion = 0

        # Append the results to the summary data
        summary_data.append({
            'Directory': sub_dir,
            'Total Good OCR Predictions': total_good_ocr_corrections,
            'Total Bad OCR Predictions': total_bad_ocr_corrections,
            'Proportion of Good OCR Predictions (%)': round(good_ocr_proportion, 2),
            'Proportion of Bad OCR Predictions (%)': round(bad_ocr_proportion, 2)
        })

# Create a DataFrame to display the results in table format
df_summary = pd.DataFrame(summary_data)

# Print the DataFrame with all rows and columns displayed
print(df_summary.to_string(index=False))
