#!/usr/bin/env python3
"""
Script to clean JSON files from OCR post-processing results.
Extracts the actual corrected text from the verbose predicted_label field.
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import shutil
from datetime import datetime


class OCRJsonCleaner:
    """Clean OCR result JSON files by extracting actual predictions."""

    def __init__(self, backup: bool = True, verbose: bool = False):
        self.backup = backup
        self.verbose = verbose
        self.stats = {
            'files_processed': 0,
            'files_cleaned': 0,
            'files_skipped': 0,
            'errors': 0,
            'predictions_cleaned': 0,
            'edge_cases': 0
        }

    def extract_corrected_text(self, predicted_label: str) -> str:
        """
        Extract the actual corrected text from the verbose predicted_label.

        Handles various patterns including:
        - Standard: 'The corrected version of the text line is: "actual text"'
        - With extra text: 'The corrected version... is: "actual text". Additional explanation...'
        - Partial: 'The corrected version... is: "actual text" Here's the breakdown...'
        - Alternative: 'The text line "X" appears... The corrected version... is likely: "Y"'
        """
        if not isinstance(predicted_label, str):
            return predicted_label

        # First, try to find patterns with "The corrected version"
        patterns = [
            # Standard patterns with various quote styles
            r'The corrected version of the text line is:\s*["\']([^"\']+)["\']',
            r'The corrected version of the text line is:\s*"([^"]+)"',
            r"The corrected version of the text line is:\s*'([^']+)'",
            r'The corrected version of the text line is:\s*«([^»]+)»',
            # Pattern with "likely" variation
            r'The corrected version of the text line is likely:\s*["\']([^"\']+)["\']',
            # Pattern that might appear in alternative formats
            r'corrected version[^:]*:\s*["\']([^"\']+)["\']',
            # Pattern for "The text line ... The corrected version..."
            r'The text line[^.]+\.\s*The corrected version[^:]*:\s*["\']([^"\']+)["\']',
        ]

        for pattern in patterns:
            match = re.search(pattern, predicted_label, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
        # Clean up any trailing punctuation that's not part of the sentence
        if extracted.endswith('."') or extracted.endswith('."'):
            extracted = extracted[:-1]
        return extracted

        # Special handling for edge cases where the format is different
        # Case: "Train, ample, the olive link set up."
        if "The corrected version" in predicted_label:
            # Try to extract text between "is:" and the next sentence or punctuation
            match = re.search(r'The corrected version[^:]*:\s*([^.!?]+)[.!?]', predicted_label)
            if match:
                extracted = match.group(1).strip()
                # Remove quotes if present
                if (extracted.startswith('"') and extracted.endswith('"')) or \
                        (extracted.startswith("'") and extracted.endswith("'")):
                    extracted = extracted[1:-1]
                return extracted

        # If the predicted_label doesn't contain our expected pattern,
        # it might already be cleaned or in a different format
        # Check if it looks like a verbose explanation
        if len(predicted_label) > 100 and ("The corrected" in predicted_label or
                                           "correction" in predicted_label or
                                           "Here's the breakdown" in predicted_label):
            self.stats['edge_cases'] += 1
            if self.verbose:
                print(f"  ⚠️  Edge case detected, unable to extract clean text")
            return predicted_label  # Return as-is for manual review

        # If it's a short string without our patterns, it might already be clean
        return predicted_label

    def clean_json_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Clean a single JSON file.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if it's a list of records
            if not isinstance(data, list):
                return False, f"File does not contain a list of records"

            # Flag to track if any changes were made
            changes_made = False
            predictions_cleaned = 0
            edge_cases = []

            # Process each record
            for idx, record in enumerate(data):
                if 'Prompt correcting' in record and isinstance(record['Prompt correcting'], dict):
                    prompt_data = record['Prompt correcting']
                    if 'predicted_label' in prompt_data:
                        original = prompt_data['predicted_label']
                        cleaned = self.extract_corrected_text(original)

                        if original != cleaned:
                            # Check if we actually extracted something meaningful
                            if len(cleaned) < len(original) / 2:  # If cleaned text is less than half the original
                                prompt_data['predicted_label'] = cleaned
                                changes_made = True
                                predictions_cleaned += 1

                                if self.verbose:
                                    print(f"  ✓ Cleaned: '{original[:50]}...' → '{cleaned}'")
                            else:
                                # This might be an edge case where extraction failed
                                edge_cases.append({
                                    'index': idx,
                                    'file_name': record.get('file_name', 'unknown'),
                                    'original': original[:100] + '...' if len(original) > 100 else original
                                })

            if edge_cases and self.verbose:
                print(f"  ⚠️  Found {len(edge_cases)} edge cases that need manual review")

            if changes_made:
                # Create backup if requested
                if self.backup:
                    backup_path = file_path.with_suffix('.json.bak')
                    shutil.copy2(file_path, backup_path)

                # Write cleaned data back
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                self.stats['predictions_cleaned'] += predictions_cleaned

                # Save edge cases report if any
                if edge_cases:
                    report_path = file_path.with_suffix('.edge_cases.json')
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(edge_cases, f, indent=2)
                    return True, f"Cleaned {predictions_cleaned} predictions, {len(edge_cases)} edge cases saved to report"

                return True, f"Cleaned {predictions_cleaned} predictions"
            else:
                return False, "No changes needed"

        except json.JSONDecodeError as e:
            return False, f"JSON decode error: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def clean_directory(self, directory: Path) -> None:
        """Clean all JSON files in a directory and its subdirectories."""
        json_files = list(directory.rglob("*.json"))

        # Filter out backup files and edge case reports
        json_files = [f for f in json_files if not (f.suffix == '.bak' or
                                                    '.bak' in str(f) or
                                                    'edge_cases' in str(f))]

        if not json_files:
            print(f"No JSON files found in {directory}")
            return

        print(f"\nFound {len(json_files)} JSON files in {directory}")
        print("=" * 80)

        for i, file_path in enumerate(json_files, 1):
            relative_path = file_path.relative_to(directory)
            print(f"\n[{i}/{len(json_files)}] Processing: {relative_path}")

            success, message = self.clean_json_file(file_path)

            if success:
                print(f"  ✓ {message}")
                self.stats['files_cleaned'] += 1
            else:
                if "No changes needed" in message:
                    print(f"  - {message}")
                    self.stats['files_skipped'] += 1
                else:
                    print(f"  ✗ {message}")
                    self.stats['errors'] += 1

            self.stats['files_processed'] += 1

    def analyze_patterns(self, directory: Path) -> None:
        """Analyze patterns in JSON files to understand the data structure."""
        print("\nAnalyzing patterns in JSON files...")
        print("=" * 80)

        json_files = list(directory.rglob("*.json"))
        json_files = [f for f in json_files if not (f.suffix == '.bak' or '.bak' in str(f))]

        patterns = defaultdict(int)
        samples = defaultdict(list)

        for file_path in json_files[:10]:  # Analyze first 10 files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for record in data[:5]:  # First 5 records per file
                        if 'Prompt correcting' in record and 'predicted_label' in record['Prompt correcting']:
                            label = record['Prompt correcting']['predicted_label']

                            # Categorize patterns
                            if 'The corrected version of the text line is:' in label:
                                if '"' in label:
                                    patterns['standard_quotes'] += 1
                                elif "'" in label:
                                    patterns['single_quotes'] += 1
                                else:
                                    patterns['no_quotes'] += 1

                                if len(label) > 200:
                                    patterns['with_explanation'] += 1

                                if 'Here\'s the breakdown' in label:
                                    patterns['with_breakdown'] += 1

                                if label.count('"') > 2:
                                    patterns['multiple_quotes'] += 1
                                    samples['multiple_quotes'].append(label[:100] + '...')
                            else:
                                patterns['non_standard'] += 1
                                samples['non_standard'].append(label[:100] + '...')

            except Exception as e:
                print(f"Error analyzing {file_path}: {str(e)}")

        print("\nPattern Analysis Results:")
        for pattern, count in patterns.items():
            print(f"  {pattern}: {count}")

        if samples:
            print("\nSample non-standard patterns:")
            for category, examples in samples.items():
                print(f"\n  {category}:")
                for ex in examples[:3]:
                    print(f"    - {ex}")

    def print_summary(self) -> None:
        """Print cleaning summary statistics."""
        print("\n" + "=" * 80)
        print("CLEANING SUMMARY")
        print("=" * 80)
        print(f"Files processed:      {self.stats['files_processed']}")
        print(f"Files cleaned:        {self.stats['files_cleaned']}")
        print(f"Files skipped:        {self.stats['files_skipped']}")
        print(f"Files with errors:    {self.stats['errors']}")
        print(f"Predictions cleaned:  {self.stats['predictions_cleaned']}")
        if self.stats['edge_cases'] > 0:
            print(f"Edge cases found:     {self.stats['edge_cases']}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean OCR result JSON files by extracting actual predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean all JSON files in results directory
  %(prog)s results/llm/

  # Clean without creating backups
  %(prog)s results/llm/ --no-backup

  # Clean with verbose output
  %(prog)s results/llm/ --verbose

  # Analyze patterns first
  %(prog)s results/llm/ --analyze

  # Clean multiple directories
  %(prog)s results/llm/ results/evaluation/ --verbose
        """
    )

    parser.add_argument(
        'directories',
        nargs='+',
        type=Path,
        help='Directory/directories containing JSON files to clean'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed cleaning information'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be cleaned without making changes'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze patterns in the data before cleaning'
    )

    args = parser.parse_args()

    # Initialize cleaner
    cleaner = OCRJsonCleaner(
        backup=not args.no_backup,
        verbose=args.verbose
    )

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        cleaner.backup = False

    print(f"\nOCR JSON Cleaner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backup: {'Enabled' if cleaner.backup else 'Disabled'}")

    # Process each directory
    for directory in args.directories:
        if not directory.exists():
            print(f"\nError: Directory '{directory}' does not exist")
            continue

        if not directory.is_dir():
            print(f"\nError: '{directory}' is not a directory")
            continue

        # Analyze patterns if requested
        if args.analyze:
            cleaner.analyze_patterns(directory)

        # Clean the directory
        cleaner.clean_directory(directory)

    # Print summary
    cleaner.print_summary()

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
