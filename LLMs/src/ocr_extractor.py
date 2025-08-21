#!/usr/bin/env python3
"""
OCR Cleaner for LLM Post-processing Results - In-Place Modification

This script cleans ONLY the LLM corrected prediction text (the
'predicted_label' inside the "Prompt correcting" section) in your
evaluation JSON files, removing any LLM commentary while maintaining
the original JSON structure.

**MODIFIED VERSION**: Updates JSON files in-place instead of creating new files.

Default behavior:
- Keeps the "Prompt correcting" section
- Cleans only "Prompt correcting" -> "predicted_label"
- Leaves OCR fields untouched
- **MODIFIES THE ORIGINAL JSON FILES IN-PLACE**

Directory Structure Expected:
{llm_outputs_path}/
├── {dataset}/                    # washington, bentham, iam
│   ├── {htr_model}/              # Flor_model, TrOCR_model
│   │   ├── {llm_name}/           # mistral, gpt-3.5-turbo, gpt-4o-mini, llama-3-8B-I
│   │   │   ├── {method}/         # method_0, method_1, method_2, etc.
│   │   │   │   ├── {partition}/  # train_25, train_50, train_75, train_100, valid, test
│   │   │   │   │   ├── results_{dict_name}_{timestamp}.json
"""

import os
import json
import logging
import re
import copy
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_cleaning_inplace.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OCRCleanerInPlace:
    """Clean LLM-corrected predicted labels in-place while maintaining original JSON structure."""

    def __init__(
        self,
        llm_outputs_path: str,
        create_backup: bool = True,
        drop_prompt_correcting: bool = False,
        clean_target: str = "prompt"  # 'prompt' | 'ocr' | 'both'
    ):
        """
        Initialize the OCR Cleaner for in-place modification.

        Args:
            llm_outputs_path: Path to LLM results directory
            create_backup: If True, create backup files before modifying
            drop_prompt_correcting: If True, remove the "Prompt correcting" field
            clean_target: Which field(s) to clean: 'prompt', 'ocr', or 'both'
        """
        self.llm_outputs_path = Path(llm_outputs_path)
        self.create_backup = create_backup
        self.drop_prompt_correcting = drop_prompt_correcting
        self.clean_target = clean_target

        # Statistics tracking
        self.stats = {
            'total_files_processed': 0,
            'total_files_modified': 0,
            'total_records_cleaned': 0,
            'files_by_dataset': defaultdict(int),
            'files_by_htr_model': defaultdict(int),
            'files_by_llm': defaultdict(int),
            'files_by_method': defaultdict(int),
            'records_by_partition': defaultdict(int),
            'cleaning_stats': {
                'total_cleaned_prompt': 0,
                'total_cleaned_ocr': 0,
                'patterns_matched': defaultdict(int)
            },
            'backed_up_files': [],
            'errors': []
        }

    # ---------- Cleaning logic (unchanged) ----------

    def _strip_wrapping_quotes(self, text: str) -> str:
        t = text.strip()
        if (t.startswith('"') and t.endswith('"')) or (t.startswith(""") and t.endswith(""")):
            return t[1:-1].strip()
        if (t.startswith("'") and t.endswith("'")):
            return t[1:-1].strip()
        return t

    def clean_predicted_label(self, predicted_label: str) -> str:
        """
        Clean the predicted label by removing LLM commentary and explanations.

        This targets cases like:
          - Leading prefixes: "Corrected:", "Correction:", "Output:", etc.
          - Quoted sentences followed by commentary
          - Sentences followed by "However/But/Note..." commentary
          - "should be:", "correction:", etc.

        Returns the cleaned sentence-like string.
        """
        if not predicted_label:
            return predicted_label

        # Normalize whitespace
        text = predicted_label.strip()

        # Remove obvious leading prefixes
        prefix_re = r'^\s*(?:corrected\s*[:\-]\s*|correction\s*[:\-]\s*|output\s*[:\-]\s*|prediction\s*[:\-]\s*)'
        text = re.sub(prefix_re, '', text, flags=re.IGNORECASE).strip()

        # Remove wrapping quotes if the entire content is quoted
        text = self._strip_wrapping_quotes(text)

        # If there is a newline, prefer the first non-empty line (LLMs sometimes add explanations below)
        if '\n' in text:
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), '')
            if first_line:
                text = first_line

        original_text = text

        # Pattern 1: Text ending with .' followed by commentary
        pattern1 = re.match(r"^(.*?\.')\s*(?:However|But|This|The|A more|It seems|Note)", text, re.IGNORECASE)
        if pattern1:
            self.stats['cleaning_stats']['patterns_matched']['ending_with_quote_period'] += 1
            return pattern1.group(1)

        # Pattern 2: Text ending with ." followed by commentary
        pattern2 = re.match(r'^(.*?\.\")\s*(?:However|But|This|The|A more|It seems|Note)', text, re.IGNORECASE)
        if pattern2:
            self.stats['cleaning_stats']['patterns_matched']['ending_with_period_quote'] += 1
            return pattern2.group(1)

        # Pattern 3: Quoted sentence followed by commentary
        pattern3 = re.match(r'^"([^"]+)"\s*(?:However|But|This|The|A more|It seems|Note|is)', text)
        if pattern3:
            self.stats['cleaning_stats']['patterns_matched']['quoted_text'] += 1
            return pattern3.group(1)

        # Pattern 4: Sentence ending with '.' followed by commentary
        pattern4 = re.match(r'^(.*?\.)\s*(?:However|But|This|The|A more|It seems|Note|Additionally)', text, re.IGNORECASE)
        if pattern4 and len(pattern4.group(1)) > 3:
            self.stats['cleaning_stats']['patterns_matched']['ending_with_period'] += 1
            return pattern4.group(1)

        # Pattern 5: Sentence ending with ! or ? followed by commentary
        pattern5 = re.match(r'^(.*?[!?])\s*(?:However|But|This|The|A more|It seems|Note)', text, re.IGNORECASE)
        if pattern5:
            self.stats['cleaning_stats']['patterns_matched']['ending_with_exclamation_question'] += 1
            return pattern5.group(1)

        # Pattern 6: Common commentary phrases
        commentary_phrases = [
            r'\s*(?:However|But|This|The phrase|A more likely|It seems|Note that|Additionally|Although)',
            r'\s*(?:might be|could be|appears to be|seems to be)',
            r'\s*(?:The correct|The actual|More accurately)',
            r'\s*(?:is unusual|is likely|is probably)',
            r'\s*(?:correction|interpretation|misinterpretation)'
        ]
        for phrase in commentary_phrases:
            parts = re.split(phrase, text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1 and len(parts[0].strip()) > 3:
                self.stats['cleaning_stats']['patterns_matched']['before_commentary'] += 1
                return parts[0].strip()

        # Pattern 7: "correction:", "should be:", etc.
        correction_match = re.match(
            r'^(.*?)\s*(?:correction:|corrected:|should be:|might be:)',
            text,
            re.IGNORECASE
        )
        if correction_match and len(correction_match.group(1)) > 3:
            self.stats['cleaning_stats']['patterns_matched']['before_correction'] += 1
            return correction_match.group(1).strip()

        # No change; return normalized original_text
        if len(original_text) > 150:
            # Large strings are suspicious; track that no pattern matched
            self.stats['cleaning_stats']['patterns_matched']['no_pattern'] += 1

        return original_text

    # ---------- Record processing ----------

    def clean_records(self, llm_result_data: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Clean predicted labels in LLM results while maintaining original structure.

        Returns:
            Tuple of (cleaned_records, was_modified)
        """
        cleaned_records = []
        was_modified = False

        for record in llm_result_data:
            try:
                # Validate basic fields
                required_fields = ['file_name', 'ground_truth_label', 'OCR']
                if not all(field in record for field in required_fields):
                    logger.warning(f"Skipping record due to missing fields: {record.get('file_name', 'unknown')}")
                    continue

                cleaned_record = copy.deepcopy(record)

                # Clean prompt-corrected predicted_label (default behavior)
                if self.clean_target in ('prompt', 'both'):
                    if 'Prompt correcting' in cleaned_record and isinstance(cleaned_record['Prompt correcting'], dict):
                        pc = cleaned_record['Prompt correcting']
                        raw = pc.get('predicted_label', '')
                        cleaned = self.clean_predicted_label(raw)
                        if raw != cleaned:
                            self.stats['cleaning_stats']['total_cleaned_prompt'] += 1
                            was_modified = True
                        pc['predicted_label'] = cleaned
                    else:
                        # If absent, keep record as-is (do not drop it)
                        logger.debug(f"No 'Prompt correcting' section in {cleaned_record.get('file_name', 'unknown')}")

                # Optionally clean OCR predicted_label
                if self.clean_target in ('ocr', 'both'):
                    ocr_data = cleaned_record.get('OCR', {})
                    if isinstance(ocr_data, dict):
                        raw_ocr = ocr_data.get('predicted_label', '')
                        cleaned_ocr = self.clean_predicted_label(raw_ocr)
                        if raw_ocr != cleaned_ocr:
                            self.stats['cleaning_stats']['total_cleaned_ocr'] += 1
                            was_modified = True
                        cleaned_record['OCR']['predicted_label'] = cleaned_ocr

                # Optionally drop "Prompt correcting"
                if self.drop_prompt_correcting and 'Prompt correcting' in cleaned_record:
                    del cleaned_record['Prompt correcting']
                    was_modified = True

                cleaned_records.append(cleaned_record)

            except Exception as e:
                error_msg = f"Error processing record {record.get('file_name', 'unknown')}: {str(e)}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
                continue

        return cleaned_records, was_modified

    # ---------- File processing (modified for in-place) ----------

    def process_json_file_inplace(
        self,
        file_path: Path,
        dataset: str,
        htr_model: str,
        llm_name: str,
        method_name: str,
        partition: str
    ) -> bool:
        """
        Process a single JSON file and clean predicted labels IN-PLACE.

        Returns:
            True if file was modified, False otherwise
        """
        try:
            logger.info(f"Processing: {file_path}")

            # Read and parse JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list) or not data:
                logger.warning(f"Empty or invalid JSON data in {file_path}")
                return False

            # Clean records while maintaining structure
            cleaned_records, was_modified = self.clean_records(data)

            if not cleaned_records:
                logger.warning(f"No valid records found in {file_path}")
                return False

            # Only write back if modifications were made
            if was_modified:
                # Create backup if requested
                if self.create_backup:
                    backup_path = file_path.with_suffix('.json.backup')
                    shutil.copy2(file_path, backup_path)
                    self.stats['backed_up_files'].append(str(backup_path))
                    logger.debug(f"Created backup: {backup_path}")

                # Write cleaned data back to the same file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_records, f, indent=2, ensure_ascii=False)

                logger.info(f"Modified {file_path} - cleaned {len(cleaned_records)} records")
                self.stats['total_files_modified'] += 1
            else:
                logger.info(f"No modifications needed for {file_path}")

            # Update statistics
            self.stats['total_files_processed'] += 1
            self.stats['total_records_cleaned'] += len(cleaned_records)
            self.stats['files_by_dataset'][dataset] += 1
            self.stats['files_by_htr_model'][htr_model] += 1
            self.stats['files_by_llm'][llm_name] += 1
            self.stats['files_by_method'][method_name] += 1
            self.stats['records_by_partition'][partition] += len(cleaned_records)

            return was_modified

        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in {file_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return False

    # ---------- Directory scan (modified for in-place) ----------

    def scan_and_clean_inplace(
        self,
        datasets: Optional[List[str]] = None,
        htr_models: Optional[List[str]] = None,
        partitions: Optional[List[str]] = None
    ) -> None:
        """
        Scan directory structure and clean predictions in all matching files IN-PLACE.

        Args:
            datasets: List of datasets to process (None for all)
            htr_models: List of HTR models to process (None for all)
            partitions: List of partitions to process (None for all)
        """
        logger.info(f"Starting in-place cleaning in {self.llm_outputs_path}")
        if self.create_backup:
            logger.info("Backup files will be created with .backup extension")

        if not self.llm_outputs_path.exists():
            raise FileNotFoundError(f"LLM outputs path does not exist: {self.llm_outputs_path}")

        # Default values if not specified
        if datasets is None:
            datasets = [d.name for d in self.llm_outputs_path.iterdir() if d.is_dir()]
        if htr_models is None:
            htr_models = ['Flor_model', 'TrOCR_model']
        if partitions is None:
            partitions = ['train_25', 'train_50', 'train_75', 'train_100', 'valid', 'test']

        modified_files = 0

        for dataset in datasets:
            dataset_path = self.llm_outputs_path / dataset
            if not dataset_path.exists():
                logger.warning(f"Dataset path does not exist: {dataset_path}")
                continue

            for htr_model in htr_models:
                htr_path = dataset_path / htr_model
                if not htr_path.exists():
                    logger.warning(f"HTR model path does not exist: {htr_path}")
                    continue

                # Scan all LLM directories
                for llm_dir in htr_path.iterdir():
                    if not llm_dir.is_dir():
                        continue

                    llm_name = llm_dir.name

                    # Scan all method directories
                    for method_dir in llm_dir.iterdir():
                        if not method_dir.is_dir():
                            continue

                        method_name = method_dir.name

                        # Scan partition directories
                        for partition in partitions:
                            partition_path = method_dir / partition
                            if not partition_path.exists():
                                continue

                            # Process all JSON files in partition directory
                            json_files = list(partition_path.glob("results_*.json"))

                            for json_file in json_files:
                                was_modified = self.process_json_file_inplace(
                                    json_file, dataset, htr_model, llm_name, method_name, partition
                                )

                                if was_modified:
                                    modified_files += 1

        logger.info(f"In-place cleaning completed. Modified {modified_files} files out of {self.stats['total_files_processed']} processed.")

    # ---------- Reporting (modified for in-place) ----------

    def generate_summary_report(self) -> Dict:
        """Generate a summary report of the in-place cleaning process."""
        report = {
            'cleaning_summary': {
                'total_files_processed': self.stats['total_files_processed'],
                'total_files_modified': self.stats['total_files_modified'],
                'total_records_cleaned': self.stats['total_records_cleaned'],
                'cleaning_date': datetime.now().isoformat(),
                'mode': 'in-place modification',
                'backup_created': self.create_backup,
                'files_backed_up': len(self.stats['backed_up_files'])
            },
            'breakdown_by_dataset': dict(self.stats['files_by_dataset']),
            'breakdown_by_htr_model': dict(self.stats['files_by_htr_model']),
            'breakdown_by_llm': dict(self.stats['files_by_llm']),
            'breakdown_by_method': dict(self.stats['files_by_method']),
            'breakdown_by_partition': dict(self.stats['records_by_partition']),
            'cleaning_statistics': {
                'total_prompt_labels_cleaned': self.stats['cleaning_stats']['total_cleaned_prompt'],
                'total_ocr_labels_cleaned': self.stats['cleaning_stats']['total_cleaned_ocr'],
                'cleaning_patterns_used': dict(self.stats['cleaning_stats']['patterns_matched'])
            },
            'errors': self.stats['errors']
        }

        # Save report in the LLM outputs directory
        report_file = self.llm_outputs_path / 'cleaning_report_inplace.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary report saved to: {report_file}")
        return report

    def print_warning_message(self) -> None:
        """Print warning message about in-place modification."""
        print(f"\n{'=' * 80}")
        print("WARNING: IN-PLACE MODIFICATION MODE")
        print(f"{'=' * 80}")
        print("This script will MODIFY YOUR ORIGINAL JSON FILES directly!")
        print(f"Target directory: {self.llm_outputs_path}")
        if self.create_backup:
            print("✓ Backup files will be created with .backup extension")
        else:
            print("✗ NO BACKUPS will be created (use --no-backup flag to disable backups)")
        print(f"{'=' * 80}")
        print("The following operations will be performed:")
        print(f"- Clean target: {self.clean_target}")
        print(f"- Drop 'Prompt correcting' field: {self.drop_prompt_correcting}")
        print(f"{'=' * 80}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Clean predicted labels in LLM evaluation results IN-PLACE')
    parser.add_argument('llm_outputs_path', help='Path to LLM results directory (will be modified in-place)')
    parser.add_argument('--datasets', nargs='+',
                        help='Specific datasets to process (default: all)')
    parser.add_argument('--htr-models', nargs='+',
                        help='Specific HTR models to process (default: all)')
    parser.add_argument('--partitions', nargs='+',
                        help='Specific partitions to process (default: all)')

    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup files before modification (risky!)')

    parser.add_argument('--drop-prompt-correcting', action='store_true',
                        help='Drop the "Prompt correcting" field in the output')

    parser.add_argument('--clean-target', choices=['prompt', 'ocr', 'both'], default='prompt',
                        help="Which predicted_label to clean: 'prompt' (default), 'ocr', or 'both'")

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize cleaner
        cleaner = OCRCleanerInPlace(
            llm_outputs_path=args.llm_outputs_path,
            create_backup=not args.no_backup,
            drop_prompt_correcting=args.drop_prompt_correcting,
            clean_target=args.clean_target
        )

        # Show warning message
        cleaner.print_warning_message()

        # Ask for confirmation unless --yes flag is used
        if not args.yes:
            response = input("Do you want to proceed with in-place modification? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Operation cancelled.")
                return

        # Clean predictions in-place
        cleaner.scan_and_clean_inplace(
            datasets=args.datasets,
            htr_models=args.htr_models,
            partitions=args.partitions
        )

        # Generate summary report
        report = cleaner.generate_summary_report()

        # Print summary
        print(f"\n{'=' * 60}")
        print("IN-PLACE CLEANING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Files processed: {report['cleaning_summary']['total_files_processed']}")
        print(f"Files modified: {report['cleaning_summary']['total_files_modified']}")
        print(f"Records cleaned: {report['cleaning_summary']['total_records_cleaned']}")
        if report['cleaning_summary']['backup_created']:
            print(f"Backup files created: {report['cleaning_summary']['files_backed_up']}")

        print(f"\nBreakdown by dataset:")
        for dataset, count in report['breakdown_by_dataset'].items():
            print(f"  {dataset}: {count} files")

        print(f"\nBreakdown by HTR model:")
        for model, count in report['breakdown_by_htr_model'].items():
            print(f"  {model}: {count} files")

        print(f"\nBreakdown by LLM:")
        for llm, count in report['breakdown_by_llm'].items():
            print(f"  {llm}: {count} files")

        print(f"\nBreakdown by method:")
        for method, count in report['breakdown_by_method'].items():
            print(f"  {method}: {count} files")

        print(f"\nCleaning statistics:")
        print(f"  Total prompt labels cleaned: {report['cleaning_statistics']['total_prompt_labels_cleaned']}")
        print(f"  Total OCR labels cleaned: {report['cleaning_statistics']['total_ocr_labels_cleaned']}")
        print(f"  Cleaning patterns used:")
        for pattern, count in report['cleaning_statistics']['cleaning_patterns_used'].items():
            print(f"    {pattern}: {count}")

        if report['errors']:
            print(f"\nErrors encountered: {len(report['errors'])}")
            for error in report['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(report['errors']) > 5:
                print(f"  ... and {len(report['errors']) - 5} more errors")

        logger.info("In-place cleaning completed successfully!")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        logger.info("Cleaning interrupted by user")
    except Exception as e:
        logger.error(f"Cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
