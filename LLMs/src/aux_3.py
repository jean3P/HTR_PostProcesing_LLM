import json
import os
import pandas as pd

from constants import results_llm
from utils.aux_processing import detect_immediate_repeated_words, detect_close_repeated_word_sequences

name_dataset = 'bentham'
base_dir = os.path.join(results_llm, name_dataset, 'Flor_model', 'mistral', 'method_1')

# Initialize a list to collect data for the table
summary_data = []

# Iterate over subdirectories (train_25, train_50, train_75, train_100)
for sub_dir in ['train_25', 'train_50', 'train_75', 'train_100']:
    dir_path = os.path.join(base_dir, sub_dir)
    if os.path.exists(dir_path):
        print(f"\nProcessing {sub_dir} directory:")

        # Initialize counters for good and bad corrections
        total_good_corrections = 0
        total_bad_corrections = 0

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

                        # Prompt Correcting Predicted Label
                        prompt_correcting_label = entry["Prompt correcting"]["predicted_label"]
                        prompt_repeated = detect_immediate_repeated_words(prompt_correcting_label)
                        prompt_close_repeated = detect_close_repeated_word_sequences(prompt_correcting_label)

                        # Case 1: Good prediction - duplicate appears in GT, OCR, and LLM
                        if (ground_truth_repeated or ground_truth_close_repeated) and (ocr_repeated or ocr_close_repeated):
                            if prompt_repeated or prompt_close_repeated:
                                total_good_corrections += 1

                        # Case 2: Good prediction - no duplicate in GT, duplicate in OCR, and LLM removes it
                        elif not (ground_truth_repeated or ground_truth_close_repeated) and (ocr_repeated or ocr_close_repeated):
                            if not (prompt_repeated or prompt_close_repeated):
                                total_good_corrections += 1

                        # Case 3: Bad prediction - duplicate in GT, duplicate in OCR, but not in LLM
                        elif (ground_truth_repeated or ground_truth_close_repeated) and (ocr_repeated or ocr_close_repeated):
                            if not (prompt_repeated or prompt_close_repeated):
                                total_bad_corrections += 1

                        # Case 4: Bad prediction - no duplicate in GT, duplicate in OCR, but also in LLM
                        elif not (ground_truth_repeated or ground_truth_close_repeated) and (ocr_repeated or ocr_close_repeated):
                            if prompt_repeated or prompt_close_repeated:
                                total_bad_corrections += 1

        # Calculate proportions
        total_corrections = total_good_corrections + total_bad_corrections
        if total_corrections > 0:
            good_correction_proportion = (total_good_corrections / total_corrections) * 100
            bad_correction_proportion = (total_bad_corrections / total_corrections) * 100
        else:
            good_correction_proportion = 0
            bad_correction_proportion = 0

        # Append the results to the summary data
        summary_data.append({
            'Directory': sub_dir,
            'Total Good Predictions': total_good_corrections,
            'Total Bad Predictions': total_bad_corrections,
            'Proportion of Good Predictions (%)': round(good_correction_proportion, 2),
            'Proportion of Bad Predictions (%)': round(bad_correction_proportion, 2)
        })

# Create a DataFrame to display the results in table format
df_summary = pd.DataFrame(summary_data)

# Print the DataFrame with all rows and columns displayed
print(df_summary.to_string(index=False))
