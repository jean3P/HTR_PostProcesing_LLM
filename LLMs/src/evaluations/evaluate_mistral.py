# src/evaluation/evaluate_mistral.py

from evaluations.metrics_evaluation import cer_only, wer_only
from utils.logger import setup_logger

logger = setup_logger()


def evaluate_and_correct_ocr_results_mistral(loaded_data, train_set_lines, text_processing_strategy, pipe,
                                             mistral_tokenizer, run_id):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']

        corrected_text_line, confidence, justification = text_processing_strategy.check_and_correct_text_line(
                ocr_text, pipe, mistral_tokenizer, train_set_lines
            )

        if ocr_text == corrected_text_line:
            cer_mistral = item['cer']
            wer_mistral = item['wer']
        else:
            cer_mistral = cer_only([corrected_text_line], [ground_truth_label])
            wer_mistral = wer_only([corrected_text_line], [ground_truth_label])

        results.append({
            'run_id': run_id,
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'wer': item['wer']
            },
            'Prompt correcting': {
                'predicted_label': corrected_text_line,
                'cer': cer_mistral,
                'wer': wer_mistral,
                'confidence': confidence,
                'justification': justification
            }
        })

    return results


def evaluate_and_correct_ocr_results_gpt(loaded_data, train_set_lines, text_processing_strategy, run_id, model_name, openai_token):
    """
    Generalized function to evaluate and correct OCR results using either the Mistral or GPT model.

    Args:
        loaded_data (list): A list of dictionaries containing OCR data (file_name, predicted_label, ground_truth_label, etc.).
        train_set_lines (list): A list of lines from the training set for suggestion purposes.
        text_processing_strategy (TextProcessingStrategy): The strategy for correcting OCR text (e.g., Mistral or GPT).
        *model_args: Additional arguments required by the specific model (e.g., pipeline, tokenizer for Mistral or model_name for GPT).

    Returns:
        list: A list of dictionaries with the evaluated and corrected OCR results.
    """
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']

        # For Mistral: Pass (ocr_text, train_set_lines, pipe, mistral_tokenizer)
        # For GPT: Pass (ocr_text, train_set_lines, model_name)
        corrected_text_line, confidence, justification = text_processing_strategy.check_and_correct_text_line(
            ocr_text, train_set_lines, model_name, openai_token
        )

        # Calculate CER only if the text line was corrected
        if ocr_text == corrected_text_line:
            cer_corrected = item['cer']
            wer_mistral = item['wer']
        else:
            cer_corrected = cer_only([corrected_text_line], [ground_truth_label])
            wer_mistral = wer_only([corrected_text_line], [ground_truth_label])

        # Append the result
        results.append({
            'run_id': run_id,
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'wer': item['wer']
            },
            'Prompt correcting': {
                'predicted_label': corrected_text_line,
                'cer': cer_corrected,
                'wer': wer_mistral,
                'confidence': confidence,  # Placeholder for confidence (if applicable)
                'justification': justification  # Placeholder for justification (if applicable)
            }
        })

    return results
