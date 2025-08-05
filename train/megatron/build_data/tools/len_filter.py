#!/usr/bin/env python3
"""
Extract lines with length >= len_filter from a JSONL file.
"""

import argparse
import os
import sys
from tqdm import tqdm


def count_lines(file_path: str) -> int:
    """Count the number of lines in a file efficiently."""
    with open(file_path, 'rb') as f:
        # Count newlines in binary mode for better performance
        return sum(1 for _ in f)

def process_file(input_file: str, output_file: str, len_filter: int) -> None:
    """Process the JSONL file and keep the lines with length >= len_filter."""
    print(f"Extracting lines with length >= {len_filter} from {input_file} to {output_file}...")

    total_lines = count_lines(input_file)
    print(f"Processing {input_file} with {total_lines} lines.")
    
    # Write lines with length >= len_filter to the output file
    with open(input_file, 'r', encoding='utf-8') as in_f, open(output_file, 'w', encoding='utf-8') as out_f:
        for line in tqdm(in_f, total=total_lines, desc="Processing"):
            if len(line) >= len_filter:
                out_f.write(line)
    
    print(f"Successfully wrote lines with length >= {len_filter} to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract lines with length >= len_filter from a JSONL file."
    )
    parser.add_argument("--input_file", help="Path to the input JSONL file")
    parser.add_argument("--output_file", help="Path to the output JSONL file")
    parser.add_argument("--len_filter", type=int, default=8000, help="Minimum length of lines to keep")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    process_file(args.input_file, args.output_file, args.len_filter)
    print("Done.")
    print()


if __name__ == "__main__":
    main()
