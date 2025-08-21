# repos/HTR_PostProcesing_LLM/LLMs/src/main_openrouter.py (updated version with logger only, no print statements)

import os
import sys
import argparse
import json
import uuid
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from constants import training_suggestion_path, results_llm
from llm.llm_factory import LLMFactory

# Import all PromptOR strategies
from prompts.openrouter.methods.OpenRouterTextProcessingPromptOR1 import OpenRouterTextProcessingPromptOR1
from prompts.openrouter.methods.OpenRouterTextProcessingPromptOR2 import OpenRouterTextProcessingPromptOR2
from prompts.openrouter.methods.OpenRouterTextProcessingPromptOR3 import OpenRouterTextProcessingPromptOR3
from prompts.openrouter.methods.OpenRouterTextProcessingPromptOR4 import OpenRouterTextProcessingPromptOR4

from utils.aux_processing import extract_text_lines_from_train_data
from utils.io_utils import get_latest_result_for_datasets, load_from_json, create_testing_file
from utils.logger import setup_logger


class OpenRouterOCRProcessor:
    """Main class for OCR post-processing with OpenRouter models."""

    # Available OpenRouter models
    OPENROUTER_MODELS = {
        "qwen-2.5-vl-72b": "Qwen 2.5 VL 72B",
        "phi-4": "Microsoft Phi-4",
        "internalvl3": "InternalVL3",
        "mistral": "Mistral",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "gpt-4.1-mini": "GPT 4.1 Mini",
        "claude-sonnet-4": "Claude Sonnet 4"
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration."""
        self.config = config
        self.setup_logging()
        self.strategies = self.get_available_strategies()
        self.stats = defaultdict(int)
        self.start_time = time.time()

    def setup_logging(self):
        """Setup logging configuration using the existing logger system."""
        # Create log file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.config.get('log_file', f"logs/openrouter_prompts_{timestamp}.log")

        # Use the existing setup_logger function
        self.logger = setup_logger(log_file)

        # Set log level based on verbose flag
        if self.config.get('verbose'):
            self.logger.setLevel(logging.DEBUG)

        # Log initial information
        self.logger.info("=" * 80)
        self.logger.info("OPENROUTER OCR POST-PROCESSING - PROMPT COMPARISON STUDY")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 80)

    def get_available_strategies(self) -> Dict[str, Any]:
        """Get all available processing strategies for OpenRouter."""
        return {
            'promptOR_1': OpenRouterTextProcessingPromptOR1,
            'promptOR_2': OpenRouterTextProcessingPromptOR2,
            'promptOR_3': OpenRouterTextProcessingPromptOR3,
            'promptOR_4': OpenRouterTextProcessingPromptOR4,
        }

    def evaluate_and_correct_ocr_results_openrouter(
            self,
            loaded_data: List[Dict],
            train_set_lines: List[str],
            text_processing_strategy: Any,
            run_id: str,
            llm_instance: Any,
            name_dataset: str,
            logger: Any,
            max_lines: Optional[int] = None
    ) -> List[Dict]:
        """Evaluate and correct OCR results using OpenRouter models."""
        results = []

        for idx, item in enumerate(loaded_data):
            if max_lines is not None and idx >= max_lines:
                break

            if idx > 0 and idx % 10 == 0:
                logger.info(f"Processed {idx}/{len(loaded_data)} samples")

            ocr_text = item['predicted_label']
            ground_truth_label = item['ground_truth_label']

            try:
                # Process the text line
                corrected_text_line, confidence, justification = text_processing_strategy.check_and_correct_text_line(
                    ocr_text,
                    train_set_lines,
                    llm_instance,
                    name_dataset,
                    logger
                )

                # Calculate metrics
                if ocr_text == corrected_text_line:
                    cer_corrected = item['cer']
                    wer_corrected = item['wer']
                else:
                    from evaluations.metrics_evaluation import cer_only, wer_only
                    cer_corrected = cer_only([corrected_text_line], [ground_truth_label])
                    wer_corrected = wer_only([corrected_text_line], [ground_truth_label])

                # Append result
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
                        'wer': wer_corrected,
                        'confidence': confidence,
                        'justification': justification
                    }
                })

                self.stats['samples_processed'] += 1

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                self.stats['samples_failed'] += 1
                continue

        return results

    def process_single_configuration(
            self,
            model: str,
            dataset: str,
            htr_model: str,
            prompt_method: str,
            train_size: str
    ) -> bool:
        """Process a single configuration."""

        config_id = f"{model}_{dataset}_{htr_model}_{prompt_method}_{train_size}"
        self.logger.info(f"\nProcessing configuration: {config_id}")

        try:
            # Get latest OCR results
            latest_results = get_latest_result_for_datasets(
                [model], [dataset], [train_size], htr_model
            )

            if not latest_results:
                self.logger.error(f"No OCR results found for {dataset}/{htr_model}/{train_size}")
                return False

            _, _, _, result_path = latest_results[0]

            # Load OCR data
            loaded_data = load_from_json(result_path)
            self.logger.info(f"Loaded {len(loaded_data)} OCR records from {result_path}")

            # Apply sample limit if specified
            if self.config['max_samples'] and self.config['max_samples'] > 0:
                loaded_data = loaded_data[:self.config['max_samples']]
                self.logger.info(f"Limited to {len(loaded_data)} samples")

            # For prompt comparison, we don't use training suggestions
            train_set_lines = []

            # Initialize OpenRouter LLM
            openrouter_llm = LLMFactory.get_llm(model)

            # Get processing strategy
            if prompt_method not in self.strategies:
                self.logger.error(f"Prompt method {prompt_method} not available")
                return False

            strategy_class = self.strategies[prompt_method]
            strategy = strategy_class()

            # Generate run ID
            run_id = str(uuid.uuid4())

            # Setup run-specific logging using your logger system
            log_file_path = f"logs/workflow_{dataset}_{htr_model}_{model}_{prompt_method}_{train_size}.log"
            run_logger = setup_logger(log_file_path)

            run_logger.info(
                f"=== Running {prompt_method} for '{dataset}' with '{train_size}' "
                f"| Model: {model} | Run ID: {run_id} ==="
            )

            # Process the data
            evaluation_results = self.evaluate_and_correct_ocr_results_openrouter(
                loaded_data,
                train_set_lines,
                strategy,
                run_id,
                openrouter_llm,
                dataset,
                run_logger,
                self.config.get('max_samples')
            )

            # Save results
            if not self.config['dry_run']:
                # Use 'no_suggestions' as the suggestion identifier for these experiments
                result_file = create_testing_file(
                    results_llm,
                    dataset,
                    train_size,
                    evaluation_results,
                    'no_suggestions',
                    model,
                    prompt_method,
                    htr_model
                )
                run_logger.info(f"Results saved to: {result_file}")
            else:
                run_logger.info("Dry run - results not saved")

            run_logger.info(
                f"=== Evaluation completed | Processed {len(evaluation_results)} samples | Run ID: {run_id} ==="
            )

            # Ensure all log handlers flush their buffers
            for handler in run_logger.handlers:
                handler.flush()

            self.stats['successful_runs'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Error processing configuration {config_id}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.stats['failed_runs'] += 1
            return False

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete processing pipeline."""

        self.logger.info("Starting OpenRouter Prompt Comparison Study")

        # Generate all configurations
        configurations = []
        for model in self.config['models']:
            for dataset in self.config['datasets']:
                for htr_model in self.config['htr_models']:
                    for prompt_method in self.config['prompt_methods']:
                        for train_size in self.config['train_sizes']:
                            configurations.append((
                                model, dataset, htr_model, prompt_method, train_size
                            ))

        total_configs = len(configurations)
        self.logger.info(f"Total configurations to process: {total_configs}")

        # Process each configuration
        successful = 0
        for i, config in enumerate(configurations, 1):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Configuration {i}/{total_configs}")
            self.logger.info(f"{'=' * 60}")

            if self.process_single_configuration(*config):
                successful += 1

            # Progress reporting
            if i % 5 == 0 or i == total_configs:
                progress = (i / total_configs) * 100
                elapsed = time.time() - self.start_time
                eta = (elapsed / i) * (total_configs - i) if i > 0 else 0
                self.logger.info(
                    f"\nProgress: {i}/{total_configs} ({progress:.1f}%) | "
                    f"Elapsed: {elapsed // 60:.0f}m | ETA: {eta // 60:.0f}m"
                )

        # Final statistics
        duration = time.time() - self.start_time

        summary = {
            'total_configurations': total_configs,
            'successful_runs': successful,
            'failed_runs': total_configs - successful,
            'samples_processed': self.stats['samples_processed'],
            'samples_failed': self.stats.get('samples_failed', 0),
            'duration_seconds': duration,
            'duration_formatted': f"{duration // 3600:.0f}h {(duration % 3600) // 60:.0f}m {duration % 60:.0f}s"
        }

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PROMPT COMPARISON STUDY COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Summary: {json.dumps(summary, indent=2)}")

        # Ensure all handlers flush
        for handler in self.logger.handlers:
            handler.flush()

        return summary


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""

    parser = argparse.ArgumentParser(
        description="OpenRouter OCR Post-processing - Prompt Comparison Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all 4 prompts with a single model
  %(prog)s --models phi-4 --datasets bentham --prompt-methods promptOR_1 promptOR_2 promptOR_3 promptOR_4

  # Compare prompts across multiple models
  %(prog)s --models qwen-2.5-vl-72b phi-4 gemini-2.5-pro \\
           --datasets iam bentham \\
           --prompt-methods promptOR_1 promptOR_2 promptOR_3 promptOR_4

  # Quick test with limited samples
  %(prog)s --models phi-4 --datasets bentham --prompt-methods promptOR_1 promptOR_3 \\
           --max-samples 10 --verbose

  # Full experiment
  %(prog)s --models qwen-2.5-vl-72b phi-4 gemini-2.5-pro gpt-4.1-mini claude-sonnet-4 \\
           --datasets bentham iam washington \\
           --htr-models Flor_model TrOCR_model \\
           --prompt-methods promptOR_1 promptOR_2 promptOR_3 promptOR_4 \\
           --train-sizes train_25 train_50 train_75 train_100
        """
    )

    # Model Selection
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(OpenRouterOCRProcessor.OPENROUTER_MODELS.keys()),
        default=['phi-4'],
        help='OpenRouter model(s) to use'
    )

    # Dataset Selection
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['bentham', 'iam', 'washington'],
        default=['iam'],
        help='Dataset(s) to process'
    )

    # HTR Model Selection
    parser.add_argument(
        '--htr-models',
        nargs='+',
        choices=['Flor_model', 'TrOCR_model'],
        default=['Flor_model'],
        help='HTR model(s) to use'
    )

    # Prompt Methods
    parser.add_argument(
        '--prompt-methods',
        nargs='+',
        choices=['promptOR_1', 'promptOR_2', 'promptOR_3', 'promptOR_4'],
        default=['promptOR_1', 'promptOR_2', 'promptOR_3', 'promptOR_4'],
        help='Prompt methods to compare (default: all 4 prompts)'
    )

    # Train Size Selection
    parser.add_argument(
        '--train-sizes',
        nargs='+',
        choices=['train_25', 'train_50', 'train_75', 'train_100'],
        default=['train_25'],
        help='Training size(s) to use'
    )

    # Processing Options
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to process per configuration'
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

    # Show prompt descriptions
    parser.add_argument(
        '--describe-prompts',
        action='store_true',
        help='Describe all prompt methods and exit'
    )

    return parser


def describe_prompts():
    """Print descriptions of all prompt methods."""
    # Create a temporary logger for description output
    desc_logger = setup_logger()

    desc_logger.info("\nOpenRouter Prompt Methods for OCR Post-processing")
    desc_logger.info("=" * 80)
    desc_logger.info("\nPromptOR_1 (Minimalist):")
    desc_logger.info("  - Simplest prompt with minimal guidance")
    desc_logger.info("  - Tests baseline performance without domain knowledge")
    desc_logger.info("  - 'Your task is to correct OCR errors...'")
    desc_logger.info("\nPromptOR_2 (Domain-specific):")
    desc_logger.info("  - Introduces century-specific context")
    desc_logger.info("  - Single guideline about historical language")
    desc_logger.info("  - 'Act as an {century}-century document analyst...'")
    desc_logger.info("\nPromptOR_3 (Extended guidelines):")
    desc_logger.info("  - Adds punctuation and hyphenation preservation rules")
    desc_logger.info("  - Prevents common over-correction issues")
    desc_logger.info("  - Includes 4 specific guidelines")
    desc_logger.info("\nPromptOR_4 (Full comprehensive):")
    desc_logger.info("  - Most comprehensive prompt with all 7 guidelines")
    desc_logger.info("  - Includes all rules from previous prompts plus additional constraints")
    desc_logger.info("  - Rules for duplicates, end modification, and number handling")
    desc_logger.info("=" * 80)


def main():
    """Main entry point."""
    # Create main logger
    main_logger = setup_logger("logs/openrouter_main.log")

    parser = create_parser()
    args = parser.parse_args()

    # Handle --describe-prompts
    if args.describe_prompts:
        describe_prompts()
        sys.exit(0)

    # Convert args to config dict
    config = vars(args)

    # Validate configuration
    errors = []

    # Check if OpenRouter API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY environment variable not set")

    if errors:
        main_logger.error("\nConfiguration errors:")
        for error in errors:
            main_logger.error(f"  ❌ {error}")
        sys.exit(1)

    try:
        # Initialize and run processor
        processor = OpenRouterOCRProcessor(config)
        summary = processor.run_pipeline()

        # Log final summary
        main_logger.info("\n" + "=" * 80)
        main_logger.info("PROMPT COMPARISON STUDY COMPLETED")
        main_logger.info("=" * 80)
        main_logger.info(f"✅ Successful runs: {summary['successful_runs']}/{summary['total_configurations']}")
        if summary['failed_runs'] > 0:
            main_logger.error(f"❌ Failed runs: {summary['failed_runs']}")
        main_logger.info(f"📊 Total samples processed: {summary['samples_processed']}")
        if summary.get('samples_failed', 0) > 0:
            main_logger.warning(f"⚠️  Failed samples: {summary['samples_failed']}")
        main_logger.info(f"⏱️  Duration: {summary['duration_formatted']}")
        main_logger.info("=" * 80)

        # Exit with appropriate code
        exit_code = 0 if summary['failed_runs'] == 0 else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        main_logger.warning("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        main_logger.error(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        main_logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
