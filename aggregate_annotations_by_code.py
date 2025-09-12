#!/usr/bin/env python3
"""
Aggregate Annotations by Code Name

This script scans through all subfolders in outputs/2021ABI, finds all JSON files 
starting with 'all_lm_ind', and creates a new JSON file that aggregates all 
annotations by unique code names.

For each code name, it collects:
- speaker
- transcript  
- justification
- file_name (original file name without 'all_lm_ind_')
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import re

def find_all_lm_ind_files(base_dir):
    """
    Find all JSON files starting with 'all_lm_ind' in subfolders.
    
    Args:
        base_dir: Base directory to search (e.g., 'outputs/2021ABI')
        
    Returns:
        List of tuples (file_path, file_name_without_prefix)
    """
    files_found = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist!")
        return files_found
    
    # Recursively search for files starting with 'all_lm_ind'
    for file_path in base_path.rglob("all_lm_ind_*.json"):
        # Extract the file name without the 'all_lm_ind_' prefix
        original_name = file_path.name.replace("all_lm_ind_", "")
        files_found.append((str(file_path), original_name))
    
    return files_found

def extract_annotations_from_file(file_path):
    """
    Extract annotations from a single JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of annotation dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    annotations = []
    
    # Handle both list and dict formats
    if isinstance(data, list):
        utterances = data
    elif isinstance(data, dict) and 'utterances' in data:
        utterances = data['utterances']
    else:
        print(f"Unexpected data format in {file_path}")
        return []
    
    for utterance in utterances:
        # Extract basic utterance info
        speaker = utterance.get('speaker', '')
        transcript = utterance.get('transcript', '')
        
        # Extract annotations
        utterance_annotations = utterance.get('annotations', {})
        
        # Handle different annotation formats
        if isinstance(utterance_annotations, dict):
            # Check if annotations has a 'codes' key
            if 'codes' in utterance_annotations:
                codes = utterance_annotations['codes']
            else:
                # Treat the entire annotations dict as a single code
                codes = [utterance_annotations]
        elif isinstance(utterance_annotations, list):
            codes = utterance_annotations
        else:
            continue
        
        # Process each code
        for code in codes:
            if isinstance(code, dict):
                code_name = code.get('code_name', '')
                justification = code.get('justification', '')
                
                if code_name and code_name.lower() != 'none':
                    annotations.append({
                        'speaker': speaker,
                        'transcript': transcript,
                        'justification': justification,
                        'code_name': code_name
                    })
    
    return annotations

def aggregate_annotations_by_code(files_list):
    """
    Aggregate annotations by unique code names.
    
    Args:
        files_list: List of tuples (file_path, file_name_without_prefix)
        
    Returns:
        Dictionary with code names as keys and lists of occurrences as values
    """
    code_aggregation = defaultdict(list)
    
    for file_path, file_name in files_list:
        print(f"Processing: {file_path}")
        
        annotations = extract_annotations_from_file(file_path)
        
        for annotation in annotations:
            code_name = annotation['code_name']
            
            # Add file_name to the annotation
            annotation['file_name'] = file_name
            
            # Store in aggregation
            code_aggregation[code_name].append({
                'speaker': annotation['speaker'],
                'transcript': annotation['transcript'],
                'justification': annotation['justification'],
                'file_name': annotation['file_name']
            })
    
    return dict(code_aggregation)

def save_aggregated_annotations(aggregated_data, output_file):
    """
    Save aggregated annotations to a JSON file.
    
    Args:
        aggregated_data: Dictionary with aggregated annotations
        output_file: Path to output JSON file
    """
    # Convert defaultdict to regular dict for JSON serialization
    serializable_data = {}
    
    for code_name, occurrences in aggregated_data.items():
        serializable_data[code_name] = {
            'total_occurrences': len(occurrences),
            'occurrences': occurrences
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"Aggregated annotations saved to: {output_file}")

def print_summary(aggregated_data):
    """
    Print a summary of the aggregated annotations.
    
    Args:
        aggregated_data: Dictionary with aggregated annotations
    """
    print("\n" + "="*60)
    print("ANNOTATION AGGREGATION SUMMARY")
    print("="*60)
    
    total_codes = len(aggregated_data)
    total_occurrences = sum(len(occurrences) for occurrences in aggregated_data.values())
    
    print(f"Total unique code names: {total_codes}")
    print(f"Total annotation occurrences: {total_occurrences}")
    
    print(f"\nTop 10 most frequent codes:")
    sorted_codes = sorted(aggregated_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (code_name, occurrences) in enumerate(sorted_codes[:10], 1):
        print(f"{i:2d}. {code_name}: {len(occurrences)} occurrences")
    
    print(f"\nAll codes and their frequencies:")
    for code_name, occurrences in sorted(aggregated_data.items()):
        print(f"  {code_name}: {len(occurrences)} occurrences")

def main():
    """
    Main function to run the annotation aggregation.
    """
    # Configuration
    base_dir = "../gemini_code/outputs/2021ABI"
    output_file = "aggregated_annotations_by_code_2021ABI.json"
    
    print("Starting annotation aggregation...")
    print(f"Searching in: {base_dir}")
    
    # Find all relevant files
    files_list = find_all_lm_ind_files(base_dir)
    
    if not files_list:
        print("No files found starting with 'all_lm_ind'!")
        return
    
    print(f"Found {len(files_list)} files to process:")
    for file_path, file_name in files_list:
        print(f"  - {file_name}")
    
    # Aggregate annotations
    print(f"\nAggregating annotations...")
    aggregated_data = aggregate_annotations_by_code(files_list)
    
    # Save results
    save_aggregated_annotations(aggregated_data, output_file)
    
    # Print summary
    print_summary(aggregated_data)
    
    print(f"\nAggregation complete!")

if __name__ == "__main__":
    main()
