import re

def extract_relevant_logs(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        capture = False
        for line in f_in:
            # Check for the start marker
            if re.search(r"=== Running for '.*' with '.*' and suggestion dictionary '.*' \| .* \| Run ID: .* ===", line):
                capture = True
            if capture:
                f_out.write(line)
            # Check for the end marker
            if re.search(r"=== Evaluation for '.*' with '.*' and suggestion dictionary '.*' completed and results saved \| .* \| Run ID: .* ===", line):
                capture = False

# Replace 'workflow.log' with your actual log file name
# Replace 'filtered_workflow.log' with your desired output file name
extract_relevant_logs('workflow.log', 'filtered_workflow.log')
