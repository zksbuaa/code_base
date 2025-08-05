#!/usr/bin/env python
"""
Parquet to JSONL converter tool

This script reads a Parquet file and outputs its contents in JSONL format.
It can be used as a standalone tool or integrated with a bash script.

Usage:
  ./parquet2jsonl.py input.parquet > output.jsonl
  or
  ./parquet2jsonl.py input.parquet output.jsonl
"""

import sys
import pyarrow.parquet as pq
from tqdm import tqdm

def parquet_to_jsonl(parquet_file, output_file=None):
    """
    Convert a Parquet file to JSONL format
    
    Args:
        parquet_file (str): Path to the Parquet file
        output_file (str, optional): Path to the output JSONL file.
                                    If None, output to stdout.
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Read the Parquet file
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # If output_file is specified, write to that file
        # Otherwise write to stdout
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting to JSONL"):
                    json_line = row.to_json(force_ascii=False)
                    outfile.write(json_line + '\n')
        else:
            for _, row in df.iterrows():
                json_line = row.to_json(force_ascii=False)
                print(json_line)
                
        return True
    except Exception as e:
        print(f"Error processing file {parquet_file}: {e}", file=sys.stderr)
        return False

def main():
    """Main function to handle CLI usage"""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print("Usage:")
        print(f"  {sys.argv[0]} input.parquet > output.jsonl")
        print(f"  {sys.argv[0]} input.parquet output.jsonl")
        return 1
    
    parquet_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = parquet_to_jsonl(parquet_file, output_file)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())