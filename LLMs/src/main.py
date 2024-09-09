# src/main.py

import os
from constants import results_from_model_path, training_suggestion_path, results_llm
from evaluations.evaluate_mistral import evaluate_and_correct_ocr_results
from llm.llm_factory import LLMFactory
from prompts.methods.mistral_text_processing_m2 import MistralTextProcessingM2
from utils.io_utils import get_latest_result_for_datasets, load_from_json, create_testing_file
from utils.logger import setup_logger

logger = setup_logger()


# Load the Mistral model using LLMFactory
mistral_llm = LLMFactory.get_llm("mistralai/Mistral-7B-v0.1")
pipe = mistral_llm.pipe
mistral_tokenizer = mistral_llm.tokenizer
llms = ['mistral']
datasets = ['iam']
train_sizes = ['train_25', 'train_50', 'train_75', 'train_100']
train_suggestion = ['bentham', 'iam', 'washington', 'whitefield', '']
latest_results = get_latest_result_for_datasets(llms, datasets, train_sizes)

for llm, dataset, train_size, result_path in latest_results:
    logger.info(f"Latest result file for {dataset}, {train_size}: {result_path}")

    # Load the latest OCR result data (assuming it's a JSON file)
    loaded_data = load_from_json(result_path)

    # Load the corresponding training suggestion data (assuming these are JSON files)
    for suggestion_file in train_suggestion:
        if suggestion_file == '':
            train_set_lines = ''
        else:
            suggestion_file_path = os.path.join(training_suggestion_path, f"{suggestion_file}.json")
            train_set_lines = load_from_json(suggestion_file_path)

        # Initialize the MistralTextProcessingM2 strategy
        text_processing_strategy = MistralTextProcessingM2()

        # Run the evaluation and correction process
        evaluation_results = evaluate_and_correct_ocr_results(
            loaded_data,
            train_set_lines,
            pipe,
            mistral_tokenizer,
            text_processing_strategy
        )

        # Save the results using create_testing_file
        create_testing_file(results_llm, dataset, train_size, evaluation_results, suggestion_file)

        logger.info(f"=== Evaluation for {dataset} with {train_size} and suggestion training set {suggestion_file} "
                    f"completed and results saved ===")

