import os

from constants import training_suggestion_path, results_llm
from evaluations.evaluate_mistral import evaluate_and_correct_ocr_results_gpt
from llm.llm_factory import LLMFactory
from prompts.gpt.methods.GptTextProcessingM1 import GptTextProcessingM1
from prompts.gpt.methods.GptTextProcessingM2 import GptTextProcessingM2
from utils.aux_processing import extract_text_lines_from_train_data
from utils.io_utils import get_latest_result_for_datasets, load_from_json, create_testing_file
from utils.logger import setup_logger
import uuid

logger = setup_logger()

llm_name_2 = "gpt-3.5-turbo"
gpt_llm = LLMFactory.get_llm(llm_name_2)
mistral_tokenizer = gpt_llm.tokenizer
llms = ['gpt-3.5']
model_ocr = "Flor_model"
name_dataset = 'bentham'
datasets = [name_dataset]
train_sizes = ['train_25', 'train_50', 'train_75', 'train_100']
train_suggestion = ['', 'bentham']
latest_results = get_latest_result_for_datasets(llms, datasets, train_sizes, model_ocr)

# Get the latest OCR results for the GPT model
latest_results = get_latest_result_for_datasets(llms, datasets, train_sizes, model_ocr)

# Define the strategies you want to iterate over
text_processing_strategies = [
    GptTextProcessingM1(),
    GptTextProcessingM2()
]

for llm, dataset, train_size, result_path in latest_results:
    logger.info(f"Latest result file for {dataset}, {train_size}: {result_path}")

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

        # Initialize the GPTTextProcessingM1 strategy
        for text_processing_strategy in text_processing_strategies:
            text_processing_strategy.suggestions_memory.clear()
            run_id = str(uuid.uuid4())

            if suggestion_file == '':
                dict_suggestion = 'empty'
            else:
                dict_suggestion = suggestion_file

            logger.info(
                f"=== Running for '{dataset}' with '{train_size}' and suggestion dictionary '{dict_suggestion}' "
                f"| {text_processing_strategy.get_name_method()} | Run ID: {run_id} ===")        # Run the evaluation and correction process
            evaluation_results = evaluate_and_correct_ocr_results_gpt(
                loaded_data,
                train_set_lines,
                text_processing_strategy,
                run_id,
                gpt_llm.model_name,  # Only pass model_name for GPT
                gpt_llm.openai_token
            )

            # Save the results using create_testing_file
            create_testing_file(results_llm, dataset, train_size, evaluation_results, suggestion_file, llm,
                                text_processing_strategy.get_name_method(), model_ocr)

            logger.info(
                f"=== Evaluation for '{dataset}' with '{train_size}' and suggestion dictionary '{dict_suggestion}' "
                f"completed and results saved | {text_processing_strategy.get_name_method()} | Run ID: {run_id} ===")
