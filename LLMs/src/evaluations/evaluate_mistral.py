# src/evaluation/evaluate_mistral.py
import os
from evaluations.metrics_evaluation import cer_only
from utils.io_utils import save_to_json
from utils.logger import setup_logger

logger = setup_logger()


def evaluate_and_correct_ocr_results(loaded_data, train_set_lines, pipe,
                                     mistral_tokenizer, text_processing_strategy):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']

        corrected_text_line, confidence, justification = text_processing_strategy.check_and_correct_text_line(
                ocr_text, pipe, mistral_tokenizer, train_set_lines
            )

        if ocr_text == corrected_text_line:
            cer_mistral = item['cer']
        else:
            cer_mistral = cer_only([corrected_text_line], [ground_truth_label])

        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
            },
            'Prompt correcting': {
                'predicted_label': corrected_text_line,
                'cer': cer_mistral,
                'confidence': confidence,
                'justification': justification
            }
        })

    return results
