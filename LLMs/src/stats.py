import json
import os

from constants import results_llm


# Function to calculate stats
def calculate_stats(data):
    total_ocr_cer = 0
    total_prompt_correcting_cer = 0
    total_cer_reduction = 0
    total_entries = len(data)

    for entry in data:
        ocr_cer = entry["OCR"]["cer"]
        prompt_correcting_cer = entry["Prompt correcting"]["cer"]

        total_ocr_cer += ocr_cer
        total_prompt_correcting_cer += prompt_correcting_cer

        # Calculate the CER reduction for each entry
        if ocr_cer > 0:  # To avoid division by zero
            cer_reduction = ((ocr_cer - prompt_correcting_cer) / ocr_cer) * 100
        else:
            cer_reduction = 0

        total_cer_reduction += cer_reduction

    mean_ocr_cer = round((total_ocr_cer / total_entries), 3)
    mean_prompt_correcting_cer = round((total_prompt_correcting_cer / total_entries), 3)
    mean_cer_reduction = round((total_cer_reduction / total_entries), 3)

    stats = {
        "mean_ocr_cer_percentage": mean_ocr_cer,
        "mean_prompt_correcting_cer_percentage": mean_prompt_correcting_cer,
        "mean_cer_reduction_percentage": mean_cer_reduction
    }

    return stats


# Function to generate LaTeX table
def generate_latex_table(stats_per_file):
    latex_code = r"""
    \subsection{Results using Washington:}

    \begin{table}[H]
        \centering
        \caption{OCR Model Comparison with Mistral LLM Across Training Scenarios}
        \label{tab:ocr_detailed_results}
        \resizebox{\textwidth}{!}{%
        \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        \textbf{Train (\%)} & \textbf{Type of Test} & \textbf{Model} & \textbf{OCR CER (\%)} & \multicolumn{2}{c|}{\textbf{MISTRAL}} \\
        \cline{5-6}
         &  &  &  & \textbf{CER (\%)} & \textbf{CER Reduction (\%)} \\
        \hline
    """

    for train_percentage, stats in stats_per_file.items():
        latex_code += f"""
        \multirow{{2}}{{*}}{{{train_percentage}}} & \multirow{{2}}{{*}}{{Final Test}} & TrOCR & {stats['mean_ocr_cer_percentage']} & {stats['mean_prompt_correcting_cer_percentage']} & {stats['mean_cer_reduction_percentage']} \\
        \cline{{3-6}}
         &  & HTR-Flor & {stats['mean_ocr_cer_percentage']} & {stats['mean_prompt_correcting_cer_percentage']} & {stats['mean_cer_reduction_percentage']} \\
        \hline
        \hline
        """

    latex_code += r"""
        \end{tabular}
        }
    \end{table}
    """
    return latex_code


# Load JSON data from file
def load_json_file(file_path):
    with (open(file_path, 'r') as file):
        data = json.load(file)
    return data


# Path to your JSON files
source = "washington"
output_path = os.path.join("..", "output", source, "flor")
directory_path_mistral = os.path.join(results_llm, 'washington', 'Flor_model', 'mistral', 'method_1', 'train_25')

# File names for each training percentage
training_percentages = ["25"]
stats_per_file = {}

# Process each file and calculate stats
for percentage in training_percentages:
    # results_bentham_2024-09-14_15-10-47.json, results_empty_2024-09-14_15-15-23.json
    json_file_path = os.path.join(directory_path_mistral, f'results_bentham_2024-09-14_15-10-47.json')

    if os.path.exists(json_file_path):
        data = load_json_file(json_file_path)
        stats = calculate_stats(data)
        stats_per_file[percentage + "\%"] = stats
    else:
        print(f"File not found: {json_file_path}")

# Generate LaTeX code for the table
latex_table_code = generate_latex_table(stats_per_file)

# Print stats and LaTeX code
print(json.dumps(stats_per_file, indent=4))
print(latex_table_code)
