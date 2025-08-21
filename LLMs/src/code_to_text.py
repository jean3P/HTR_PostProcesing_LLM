#!/usr/bin/env python3
"""
Python Code Combiner Script
Combines all Python files from a source directory into a single text file,
with options to exclude specific directories.
"""

import os
import argparse
import sys
from pathlib import Path


def should_exclude_directory(dir_path, exclusions):
    """
    Check if a directory should be excluded based on exclusion patterns.

    Args:
        dir_path (str): Path to the directory to check
        exclusions (list): List of directory names/patterns to exclude

    Returns:
        bool: True if directory should be excluded, False otherwise
    """
    dir_name = os.path.basename(dir_path)
    return dir_name in exclusions


def collect_python_files(source_dir, exclusions):
    """
    Recursively collect all Python files from source directory,
    excluding specified directories.

    Args:
        source_dir (str): Source directory path
        exclusions (list): List of directories to exclude

    Returns:
        list: List of Python file paths
    """
    python_files = []

    for root, dirs, files in os.walk(source_dir):
        # Remove excluded directories from dirs list to prevent os.walk from entering them
        dirs[:] = [d for d in dirs if not should_exclude_directory(os.path.join(root, d), exclusions)]

        # Collect Python files from current directory
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return sorted(python_files)


def combine_python_files(source_dir, exclusions, output_file):
    """
    Combine all Python files into a single text file.

    Args:
        source_dir (str): Source directory path
        exclusions (list): List of directories to exclude
        output_file (str): Output file path
    """
    python_files = collect_python_files(source_dir, exclusions)

    if not python_files:
        print("No Python files found in the specified directory.")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(f"# Combined Python Code from: {source_dir}\n")
            outfile.write(f"# Excluded directories: {', '.join(exclusions) if exclusions else 'None'}\n")
            outfile.write(f"# Total files combined: {len(python_files)}\n")
            outfile.write("# " + "=" * 80 + "\n\n")

            for i, file_path in enumerate(python_files, 1):
                relative_path = os.path.relpath(file_path, source_dir)

                # Write file header
                outfile.write(f"\n# File {i}/{len(python_files)}: {relative_path}\n")
                outfile.write("# " + "-" * 80 + "\n\n")

                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)

                    # Add separator between files
                    outfile.write(f"\n\n# End of {relative_path}\n")
                    outfile.write("# " + "=" * 80 + "\n")

                except UnicodeDecodeError:
                    outfile.write(f"# Error: Could not read file {relative_path} (encoding issue)\n")
                    print(f"Warning: Could not read {file_path} due to encoding issues")
                except Exception as e:
                    outfile.write(f"# Error reading file {relative_path}: {str(e)}\n")
                    print(f"Warning: Error reading {file_path}: {str(e)}")

        print(f"Successfully combined {len(python_files)} Python files into '{output_file}'")

    except Exception as e:
        print(f"Error writing to output file: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Combine all Python files from a source directory into a single text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source-dir /path/to/project --output combined_code.txt
  %(prog)s --source-dir ./my_project --exclude-dirs __pycache__ .git venv --output all_code.txt
  %(prog)s -s ./src -e tests docs build -o project_code.txt
        """
    )

    parser.add_argument(
        '--source-dir', '-s',
        required=True,
        help='Source directory containing Python files'
    )

    parser.add_argument(
        '--exclude-dirs', '-e',
        nargs='*',
        default=['__pycache__', '.git', '.vscode', 'venv', 'env', '.env', 'node_modules'],
        help='Directories to exclude (default: __pycache__ .git .vscode venv env .env node_modules)'
    )

    parser.add_argument(
        '--output', '-o',
        default='combined_python_code.txt',
        help='Output file name (default: combined_python_code.txt)'
    )

    args = parser.parse_args()

    # Validate source directory
    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist or is not a directory")
        sys.exit(1)

    # Convert to absolute path
    source_dir = os.path.abspath(args.source_dir)

    print(f"Source directory: {source_dir}")
    print(f"Excluded directories: {args.exclude_dirs}")
    print(f"Output file: {args.output}")
    print("-" * 50)

    combine_python_files(source_dir, args.exclude_dirs, args.output)


if __name__ == "__main__":
    main()
