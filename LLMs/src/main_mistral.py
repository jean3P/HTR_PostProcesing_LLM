# src/main_mistral.py

import os
from constants import training_suggestion_path, results_llm
from evaluations.evaluate_mistral import evaluate_and_correct_ocr_results_mistral
from llm.llm_factory import LLMFactory
from prompts.mistral.methods.mistral_text_processing_m1 import MistralTextProcessingM1
from prompts.mistral.methods.mistral_text_processing_m2 import MistralTextProcessingM2
from utils.aux_processing import extract_text_lines_from_train_data
from utils.io_utils import get_latest_result_for_datasets, load_from_json, create_testing_file
from utils.logger import setup_logger
import uuid

llm_name_1 = "mistralai/Mistral-7B-v0.1"
mistral_llm = LLMFactory.get_llm(llm_name_1)
pipe = mistral_llm.pipe
mistral_tokenizer = mistral_llm.tokenizer
llms = ['mistral']
model_ocr = "Flor_model"
datasets = ['iam']
train_sizes = ['train_25', 'train_50', 'train_75', 'train_100']
train_suggestion = ['', 'iam']
latest_results = get_latest_result_for_datasets(llms, datasets, train_sizes, model_ocr)

# Define the strategies you want to iterate over
text_processing_strategies = [
    MistralTextProcessingM1(),
]

for llm, dataset, train_size, result_path in latest_results:
    # Load the latest OCR result data (assuming it's a JSON file)
    loaded_data = load_from_json(result_path)

    # Load the corresponding training suggestion data (assuming these are JSON files)
    for suggestion_file in train_suggestion:
        if suggestion_file == '':
            train_set_lines = ''
            dict_suggestion = 'empty'
        else:
            dict_suggestion = suggestion_file
            suggestion_file_path = os.path.join(training_suggestion_path, f"{suggestion_file}.json")
            train_set_lines = load_from_json(suggestion_file_path)
            train_set_lines = extract_text_lines_from_train_data(train_set_lines)

        for text_processing_strategy in text_processing_strategies:
            text_processing_strategy.suggestions_memory.clear()
            run_id = str(uuid.uuid4())

            log_file_path = f"./logs/workflow_{dataset}_{model_ocr}_{llm}_{text_processing_strategy.get_name_method()}_{train_size}_{dict_suggestion}.log"
            print(f"Log file path: {log_file_path}")
            logger = setup_logger(log_file_path)

            logger.info(
                f"=== Running for '{dataset}' with '{train_size}' and suggestion dictionary '{dict_suggestion}' "
                f"| {text_processing_strategy.get_name_method()} | Run ID: {run_id} ===")
            for handler in logger.handlers:
                handler.flush()
            # Run the evaluation and correction process
            evaluation_results = evaluate_and_correct_ocr_results_mistral(
                loaded_data,
                train_set_lines,
                text_processing_strategy,  # Pass the strategy directly
                pipe,  # Pass pipe as an additional model argument
                mistral_tokenizer,  # Pass tokenizer as an additional model argument
                run_id,
                logger
            )

            # Save the results using create_testing_file
            create_testing_file(results_llm, dataset, train_size, evaluation_results, suggestion_file, llm,
                                text_processing_strategy.get_name_method(), model_ocr)

            logger.info(
                f"=== Evaluation for '{dataset}' with '{train_size}' and suggestion dictionary '{dict_suggestion}' "
                f"completed and results saved | {text_processing_strategy.get_name_method()} | Run ID: {run_id} ===")
