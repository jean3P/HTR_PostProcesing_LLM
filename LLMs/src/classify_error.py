import json
import difflib
import os
import random

from constants import results_llm

NUMBER_ = 100


def classify_error(ground_truth, prediction):
    """
    Classify the type of error by comparing ground truth and predicted text.
    Returns a dictionary with percentages for each error type.
    """
    if ground_truth == prediction:
        return {"Deletion": 0.0, "Substitution": 0.0, "Insertion": 0.0, "Transposition": 0.0}

    # Initialize counters for each error type
    error_counts = {"Deletion": 0, "Substitution": 0, "Insertion": 0, "Transposition": 0}
    total_edits = 0  # Total count of all edits

    # Find the differences in characters or words
    seq = difflib.SequenceMatcher(None, ground_truth, prediction)

    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        if tag == 'replace':
            error_counts["Substitution"] += 1
        elif tag == 'delete':
            error_counts["Deletion"] += 1
        elif tag == 'insert':
            error_counts["Insertion"] += 1
        elif tag == 'equal' and i1 != j1:
            error_counts["Transposition"] += 1

    total_edits = sum(error_counts.values())  # Calculate total edits

    # Convert counts to percentages
    if total_edits > 0:
        for error_type in error_counts:
            error_counts[error_type] = (error_counts[error_type] / total_edits) * 100

    return error_counts


def process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    results = []
    for entry in data:
        ground_truth = entry["ground_truth_label"]

        # Skip entries with ground truth "Sentence Database P02-109"
        if ground_truth == "Sentence Database P02-109":
            continue

        # Classify Prompt correcting errors
        prompt_prediction = entry["Prompt correcting"]["predicted_label"]
        prompt_errors = classify_error(ground_truth, prompt_prediction)
        prompt_cer = entry["Prompt correcting"].get("cer", 0.0)

        # Append results with detailed error percentages
        results.append({
            "file_name": entry["file_name"],
            "ground_truth_label": ground_truth,
            "Prompt_correcting_predicted_label": prompt_prediction,
            "Deletion (%)": prompt_errors["Deletion"],
            "Substitution (%)": prompt_errors["Substitution"],
            "Insertion (%)": prompt_errors["Insertion"],
            "Transposition (%)": prompt_errors["Transposition"],
            "Prompt_cer": prompt_cer
        })

    # Sort by Prompt CER and select top 100
    top_100_worst_prompt = sorted(results, key=lambda x: x["Prompt_cer"], reverse=True)[:NUMBER_]
    # Randomly select 100 from remaining data
    remaining_data = [item for item in results if item not in top_100_worst_prompt]
    random_100 = random.sample(remaining_data, min(NUMBER_, len(remaining_data)))

    return top_100_worst_prompt, random_100


def escape_latex_characters(text):
    """
    Escape LaTeX special characters in the provided text.
    """
    replacements = {
        '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#',
        '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}', '\\': '\\textbackslash{}'
    }
    for char, escaped_char in replacements.items():
        text = text.replace(char, escaped_char)
    return text


def print_summary_latex_table(data, caption):
    # Initialize counters for each error type
    deletion_count = 0
    substitution_count = 0
    insertion_count = 0
    transposition_count = 0

    # Count occurrences for each error type
    for entry in data:
        if entry['Deletion (%)'] > 0:
            deletion_count += 1
        if entry['Substitution (%)'] > 0:
            substitution_count += 1
        if entry['Insertion (%)'] > 0:
            insertion_count += 1
        if entry['Transposition (%)'] > 0:
            transposition_count += 1

    # Print the summary table in LaTeX format
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\scriptsize")
    print("\\begin{tabular}{|l|r|}")
    print("\\hline")
    print("\\textbf{Error Type} & \\textbf{Count} \\\\ \\hline")
    print(f"Deletion Errors & {deletion_count} \\\\ \\hline")
    print(f"Substitution Errors & {substitution_count} \\\\ \\hline")
    print(f"Insertion Errors & {insertion_count} \\\\ \\hline")
    print(f"Transposition Errors & {transposition_count} \\\\ \\hline")
    print("\\end{tabular}")
    print(f"\\caption{{{caption}}}")
    label = caption.lower().replace(' ', '_').replace('-', '_')
    print(f"\\label{{table:{label}}}")
    print("\\end{table}")


def print_latex_table(data, caption):
    print("\\newgeometry{left=1cm, right=1cm, top=0.5cm, bottom=0.5cm, landscape}")
    print("\\begin{landscape}")
    print("\\scriptsize")  # Adjust font size as needed
    print("\\begin{longtable}{|p{7cm}|p{7cm}|r|r|r|r|r|}")
    print("\\hline")
    print(
        "\\textbf{Ground Truth} & \\textbf{Prompt Prediction} & \\textbf{Deletion (\\%)} & \\textbf{Substitution (\\%)} & "
        "\\textbf{Insertion (\\%)} & \\textbf{Transposition (\\%)} & \\textbf{CER (Prompt)} \\\\ \\hline"
    )
    print("\\endfirsthead")  # For longtable, this defines the header on the first page

    # Header repeated at the top of each subsequent page
    print("\\hline")
    print(
        "\\textbf{Ground Truth} & \\textbf{Prompt Prediction} & \\textbf{Deletion (\\%)} & \\textbf{Substitution (\\%)} & "
        "\\textbf{Insertion (\\%)} & \\textbf{Transposition (\\%)} & \\textbf{CER (Prompt)} \\\\ \\hline"
    )
    print("\\endhead")

    for entry in data:
        ground_truth = escape_latex_characters(entry['ground_truth_label'])
        prompt_prediction = escape_latex_characters(entry['Prompt_correcting_predicted_label'])

        print(f"{ground_truth} & {prompt_prediction} & "
              f"{entry['Deletion (%)']:.2f} & {entry['Substitution (%)']:.2f} & "
              f"{entry['Insertion (%)']:.2f} & {entry['Transposition (%)']:.2f} & "
              f"{entry['Prompt_cer']:.2f} \\\\ \\hline")

    print("\\end{longtable}")
    print(f"\\caption{{{caption}}}")
    label = caption.lower().replace(' ', '_').replace('-', '_')
    print(f"\\label{{table:{label}}}")
    print("\\end{landscape}")
    print("\\restoregeometry")


# Example usage
base_dir = os.path.join(results_llm, 'iam', 'Flor_model', 'mistral', 'method_1', 'train_100')
file_path = os.path.join(base_dir, 'results_empty_2024-09-23_07-12-07.json')

# Process JSON and get both datasets
top_100_worst_prompt, random_100 = process_json(file_path)

# Print LaTeX tables for both datasets
print_latex_table(top_100_worst_prompt, "Error classification of Prompt - Top 100 Worst Cases")

print_summary_latex_table(top_100_worst_prompt, 'Summary of Error Types - Top 100 Worst Cases')
print_summary_latex_table(random_100, 'Summary of Error Types - Randomly Selected 100 Cases')

print_latex_table(random_100, "Error classification of Prompt - Randomly Selected 100 Cases")
