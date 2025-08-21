#!/usr/bin/env python3
"""
Professional OCR Post-processing Pipeline with LLM Correction

This script provides a comprehensive interface for correcting OCR errors using
various Large Language Models (GPT, Mistral) with multiple correction strategies.

Author: Your Name
Version: 1.0.0
"""

import argparse
import sys
import os
import json
import uuid
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging
from datetime import datetime
from collections import defaultdict

# Import your existing modules
from constants import training_suggestion_path, results_llm
from evaluations.evaluate_mistral import evaluate_and_correct_ocr_results_mistral, evaluate_and_correct_ocr_results_gpt
from llm.llm_factory import LLMFactory
from utils.aux_processing import extract_text_lines_from_train_data
from utils.io_utils import load_from_json, create_testing_file, get_latest_result_for_datasets
from utils.logger import setup_logger

# Import all available strategies
from prompts.gpt.methods.GptTextProcessingM0 import GptTextProcessingM0
from prompts.gpt.methods.GptTextProcessingM1 import GptTextProcessingM1
from prompts.gpt.methods.GptTextProcessingM2 import GptTextProcessingM2
from prompts.gpt.methods.GptTextProcessingM1P1 import GptTextProcessingM1P1
from prompts.gpt.methods.GptTextProcessingM1P2 import GptTextProcessingM1P2
from prompts.gpt.methods.GptTextProcessingM1P3 import GptTextProcessingM1P3
from prompts.gpt.methods.GptTextProcessing_m1_paper import GptTextProcessingM1_V1

from prompts.mistral.methods.mistral_text_processing_m0 import MistralTextProcessingM0
from prompts.mistral.methods.mistral_text_processing_m1 import MistralTextProcessingM1
from prompts.mistral.methods.mistral_text_processing_m2 import MistralTextProcessingM2
from prompts.mistral.methods.mistral_text_processing_m1_p1 import MistralTextProcessingM1P1
from prompts.mistral.methods.mistral_text_processing_m1_p2 import MistralTextProcessingM1P2
from prompts.mistral.methods.mistral_text_processing_m1_p3 import MistralTextProcessingM1P3


class OCRPostProcessor:
    """Main class for OCR post-processing with LLMs."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration."""
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.strategies = self.get_available_strategies()
        self.stats = defaultdict(int)

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config['verbose'] else logging.INFO
        log_file = self.config.get('log_file', f"ocr_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        self.logger = setup_logger(log_file)
        self.logger.setLevel(log_level)

        self.logger.info("=" * 80)
        self.logger.info("OCR POST-PROCESSING PIPELINE STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

    def setup_device(self):
        """Setup device configuration for GPU/CPU usage."""
        if self.config['device'] == 'auto':
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = self.config['device']

        self.logger.info(f"Using device: {self.device}")

        # Set environment variables for device usage
        if self.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get all available processing strategies."""
        return {
            'gpt': {
                'method_0': GptTextProcessingM0,
                'method_1': GptTextProcessingM1,
                'method_2': GptTextProcessingM2,
                'method_1_V1': GptTextProcessingM1P1,
                'method_1_V2': GptTextProcessingM1P2,
                'method_1_V3': GptTextProcessingM1P3,
                'method_1_paper': GptTextProcessingM1_V1,
            },
            'mistral': {
                'method_0': MistralTextProcessingM0,
                'method_1': MistralTextProcessingM1,
                'method_2': MistralTextProcessingM2,
                'method_1_V1': MistralTextProcessingM1P1,
                'method_1_V2': MistralTextProcessingM1P2,
                'method_1_V3': MistralTextProcessingM1P3,
            }
        }

    def load_ocr_data(self, dataset: str, htr_model: str, llm_name: str,
                      method: str, train_size: str) -> Optional[List[Dict]]:
        """Load OCR data from either original results or extracted OCR files."""

        if self.config['use_extracted_ocr']:
            return self.load_extracted_ocr_data(dataset, htr_model, llm_name, method, train_size)
        else:
            return self.load_original_ocr_data(dataset, htr_model, train_size)

    def load_extracted_ocr_data(self, dataset: str, htr_model: str, llm_name: str,
                                method: str, train_size: str) -> Optional[List[Dict]]:
        """Load data from extracted OCR JSON files."""

        # Look for OCR files in the results directory
        ocr_pattern = f"ocr_{dataset}_{htr_model}_{llm_name}_{method}_{train_size}_*.json"
        search_dir = Path(results_llm) / dataset / htr_model / llm_name / method / train_size

        if not search_dir.exists():
            self.logger.warning(f"OCR directory not found: {search_dir}")
            return None

        ocr_files = list(search_dir.glob(f"ocr_{dataset}_{htr_model}_{llm_name}_{method}_{train_size}_*.json"))

        if not ocr_files:
            self.logger.warning(f"No OCR files found matching pattern: {ocr_pattern}")
            return None

        # Use the most recent file
        ocr_file = sorted(ocr_files)[-1]
        self.logger.info(f"Loading extracted OCR data from: {ocr_file}")

        try:
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            # Convert extracted OCR format to expected format
            converted_data = []
            for record in ocr_data:
                converted_record = {
                    'file_name': record['file_name'],
                    'ground_truth_label': record['ground_truth'],
                    'predicted_label': record['ocr_prediction'],
                    'cer': record['ocr_cer'],
                    'wer': record['ocr_wer']
                }
                converted_data.append(converted_record)

            self.logger.info(f"Loaded {len(converted_data)} OCR records from extracted file")
            return converted_data

        except Exception as e:
            self.logger.error(f"Error loading extracted OCR file {ocr_file}: {str(e)}")
            return None

    def load_original_ocr_data(self, dataset: str, htr_model: str, train_size: str) -> Optional[List[Dict]]:
        """Load data from original OCR result files."""

        # Use existing function to get latest results
        latest_results = get_latest_result_for_datasets(
            ['dummy'], [dataset], [train_size], htr_model
        )

        if not latest_results:
            self.logger.warning(f"No original OCR results found for {dataset}/{htr_model}/{train_size}")
            return None

        _, _, _, result_path = latest_results[0]
        self.logger.info(f"Loading original OCR data from: {result_path}")

        try:
            data = load_from_json(result_path)
            self.logger.info(f"Loaded {len(data)} OCR records from original file")
            return data
        except Exception as e:
            self.logger.error(f"Error loading original OCR file {result_path}: {str(e)}")
            return None

    def load_training_suggestions(self, suggestion_name: str) -> Tuple[List[str], str]:
        """Load training suggestions."""
        if suggestion_name == 'empty' or not suggestion_name:
            return [], 'empty'

        suggestion_file_path = Path(training_suggestion_path) / f"{suggestion_name}.json"

        if not suggestion_file_path.exists():
            self.logger.warning(f"Training suggestion file not found: {suggestion_file_path}")
            return [], 'empty'

        try:
            train_set_data = load_from_json(suggestion_file_path)
            train_set_lines = extract_text_lines_from_train_data(train_set_data)
            self.logger.info(f"Loaded {len(train_set_lines)} training suggestions from {suggestion_name}")
            return train_set_lines, suggestion_name
        except Exception as e:
            self.logger.error(f"Error loading training suggestions: {str(e)}")
            return [], 'empty'

    def initialize_llm(self, llm_name: str) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Initialize the specified LLM."""
        try:
            if llm_name in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini', 'gpt-4o']:
                llm = LLMFactory.get_llm(llm_name)
                return llm, None, None
            elif llm_name in ['mistral', 'mistralai/Mistral-7B-v0.1']:
                llm = LLMFactory.get_llm('mistralai/Mistral-7B-v0.1')
                return llm, llm.pipe, llm.tokenizer
            else:
                raise ValueError(f"Unsupported LLM: {llm_name}")

        except Exception as e:
            self.logger.error(f"Error initializing LLM {llm_name}: {str(e)}")
            raise

    def process_single_configuration(self, dataset: str, htr_model: str, llm_name: str,
                                     method: str, train_size: str, train_suggestion: str) -> bool:
        """Process a single configuration."""

        config_id = f"{dataset}_{htr_model}_{llm_name}_{method}_{train_size}_{train_suggestion}"
        self.logger.info(f"Processing configuration: {config_id}")

        try:
            # Load OCR data
            ocr_data = self.load_ocr_data(dataset, htr_model, llm_name, method, train_size)
            if not ocr_data:
                self.logger.error(f"Failed to load OCR data for {config_id}")
                return False

            # Load training suggestions
            train_set_lines, dict_suggestion = self.load_training_suggestions(train_suggestion)

            # Initialize LLM
            llm, pipe, tokenizer = self.initialize_llm(llm_name)

            # Get processing strategy
            llm_type = 'gpt' if llm_name.startswith('gpt') else 'mistral'
            if method not in self.strategies[llm_type]:
                self.logger.error(f"Method {method} not available for {llm_type}")
                return False

            strategy_class = self.strategies[llm_type][method]
            strategy = strategy_class()
            strategy.suggestions_memory.clear()

            # Generate run ID
            run_id = str(uuid.uuid4())

            # Setup logging for this run
            log_file_path = f"logs/workflow_{dataset}_{htr_model}_{llm_name}_{method}_{train_size}_{dict_suggestion}.log"
            run_logger = setup_logger(log_file_path)

            run_logger.info(f"=== Starting run for '{config_id}' | Run ID: {run_id} ===")

            # Limit processing if specified
            if self.config['max_samples'] and self.config['max_samples'] > 0:
                ocr_data = ocr_data[:self.config['max_samples']]
                run_logger.info(f"Limited processing to {len(ocr_data)} samples")

            # Process based on LLM type
            if llm_type == 'gpt':
                results = evaluate_and_correct_ocr_results_gpt(
                    ocr_data, train_set_lines, strategy, run_id,
                    llm.model_name, llm.openai_token, llm_name, run_logger
                )
            else:  # mistral
                results = evaluate_and_correct_ocr_results_mistral(
                    ocr_data, train_set_lines, strategy, pipe, tokenizer, run_id, run_logger
                )

            # Save results
            if not self.config['dry_run']:
                result_file = create_testing_file(
                    results_llm, dataset, train_size, results,
                    dict_suggestion, llm_name, method, htr_model
                )
                run_logger.info(f"Results saved to: {result_file}")
            else:
                run_logger.info("Dry run - results not saved")

            run_logger.info(f"=== Completed run for '{config_id}' | Run ID: {run_id} ===")

            self.stats['successful_runs'] += 1
            self.stats['total_samples_processed'] += len(results)

            return True

        except Exception as e:
            self.logger.error(f"Error processing configuration {config_id}: {str(e)}")
            self.stats['failed_runs'] += 1
            return False

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete processing pipeline."""

        start_time = time.time()
        self.logger.info("Starting OCR post-processing pipeline")

        # Generate all configurations
        configurations = []
        for dataset in self.config['datasets']:
            for htr_model in self.config['htr_models']:
                for llm_name in self.config['llms']:
                    for method in self.config['methods']:
                        for train_size in self.config['train_sizes']:
                            for train_suggestion in self.config['train_suggestions']:
                                configurations.append((
                                    dataset, htr_model, llm_name, method, train_size, train_suggestion
                                ))

        total_configs = len(configurations)
        self.logger.info(f"Total configurations to process: {total_configs}")

        # Process each configuration
        successful = 0
        for i, config in enumerate(configurations, 1):
            self.logger.info(f"Processing configuration {i}/{total_configs}")

            if self.process_single_configuration(*config):
                successful += 1

            # Progress reporting
            if i % 10 == 0 or i == total_configs:
                progress = (i / total_configs) * 100
                self.logger.info(f"Progress: {i}/{total_configs} ({progress:.1f}%)")

        # Final statistics
        end_time = time.time()
        duration = end_time - start_time

        summary = {
            'total_configurations': total_configs,
            'successful_runs': successful,
            'failed_runs': total_configs - successful,
            'total_samples_processed': self.stats['total_samples_processed'],
            'duration_seconds': duration,
            'duration_formatted': f"{duration // 3600:.0f}h {(duration % 3600) // 60:.0f}m {duration % 60:.0f}s"
        }

        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Summary: {json.dumps(summary, indent=2)}")

        return summary


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""

    parser = argparse.ArgumentParser(
        description="Professional OCR Post-processing Pipeline with LLM Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single options
  %(prog)s --llms gpt-3.5-turbo --datasets bentham --methods method_1

  # Multiple options
  %(prog)s --llms gpt-3.5-turbo gpt-4o-mini --datasets bentham iam --methods method_1 method_2

  # Full pipeline with all options
  %(prog)s --llms gpt-3.5-turbo mistral --datasets bentham iam washington \\
           --htr-models Flor_model TrOCR_model --methods method_1 method_2 \\
           --train-sizes train_25 train_50 --train-suggestions empty bentham

  # Use extracted OCR files instead of original results
  %(prog)s --llms gpt-3.5-turbo --datasets bentham --use-extracted-ocr

  # Dry run to test configuration
  %(prog)s --llms gpt-3.5-turbo --datasets bentham --dry-run

  # GPU processing with verbose logging
  %(prog)s --llms mistral --datasets bentham --device gpu --verbose
        """
    )

    # LLM Selection
    parser.add_argument(
        '--llms',
        nargs='+',
        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini', 'gpt-4o', 'mistral'],
        default=['gpt-3.5-turbo'],
        help='LLM(s) to use for correction (default: gpt-3.5-turbo)'
    )

    # Dataset Selection
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['bentham', 'iam', 'washington'],
        default=['bentham'],
        help='Dataset(s) to process (default: bentham)'
    )

    # HTR Model Selection
    parser.add_argument(
        '--htr-models',
        nargs='+',
        choices=['Flor_model', 'TrOCR_model'],
        default=['Flor_model'],
        help='HTR model(s) to use (default: Flor_model)'
    )

    # Method Selection
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['method_0', 'method_1', 'method_2', 'method_1_V1', 'method_1_V2', 'method_1_V3', 'method_1_paper'],
        default=['method_1'],
        help='Correction method(s) to use (default: method_1)'
    )

    # Train Size Selection
    parser.add_argument(
        '--train-sizes',
        nargs='+',
        choices=['train_25', 'train_50', 'train_75', 'train_100'],
        default=['train_25'],
        help='Training size(s) to use (default: train_25)'
    )

    # Training Suggestions
    parser.add_argument(
        '--train-suggestions',
        nargs='+',
        default=['empty'],
        help='Training suggestion dataset(s) to use (default: empty). Use "empty" for no suggestions.'
    )

    # Device Selection
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu', 'auto'],
        default='auto',
        help='Device to use for processing (default: auto)'
    )

    # Input Source
    parser.add_argument(
        '--use-extracted-ocr',
        action='store_true',
        help='Use extracted OCR files instead of original result files'
    )

    # Processing Options
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to process per configuration (for testing)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without saving results'
    )

    # Logging Options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--log-file',
        help='Custom log file path'
    )

    # Output Options
    parser.add_argument(
        '--output-dir',
        default=results_llm,
        help=f'Output directory for results (default: {results_llm})'
    )

    return parser


def validate_configuration(config: Dict[str, Any]) -> List[str]:
    """Validate the configuration and return any errors."""
    errors = []

    # Check if training suggestion files exist
    for suggestion in config['train_suggestions']:
        if suggestion != 'empty':
            suggestion_file = Path(training_suggestion_path) / f"{suggestion}.json"
            if not suggestion_file.exists():
                errors.append(f"Training suggestion file not found: {suggestion_file}")

    # Check output directory
    output_dir = Path(config['output_dir'])
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory {output_dir}: {str(e)}")

    # Validate method compatibility with LLMs
    gpt_methods = ['method_0', 'method_1', 'method_2', 'method_1_V1', 'method_1_V2', 'method_1_V3', 'method_1_paper']
    mistral_methods = ['method_0', 'method_1', 'method_2', 'method_1_V1', 'method_1_V2', 'method_1_V3']

    for llm in config['llms']:
        for method in config['methods']:
            if llm.startswith('gpt') and method not in gpt_methods:
                errors.append(f"Method {method} not available for {llm}")
            elif llm == 'mistral' and method not in mistral_methods:
                errors.append(f"Method {method} not available for {llm}")

    return errors


def main():
    """Main entry point."""

    parser = create_parser()
    args = parser.parse_args()

    # Convert args to config dict
    config = vars(args)

    # Validate configuration
    errors = validate_configuration(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    try:
        # Initialize and run processor
        processor = OCRPostProcessor(config)
        summary = processor.run_pipeline()

        # Print final summary
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Processed {summary['total_configurations']} configurations")
        print(f"Successful: {summary['successful_runs']}")
        print(f"Failed: {summary['failed_runs']}")
        print(f"Total samples processed: {summary['total_samples_processed']}")
        print(f"Duration: {summary['duration_formatted']}")
        print("=" * 80)

        # Exit with appropriate code
        exit_code = 0 if summary['failed_runs'] == 0 else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
