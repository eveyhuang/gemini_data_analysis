#!/usr/bin/env python3
"""
Aggregate Annotations by Code Name

This script scans through all output folders and subfolders, finds all JSON files 
starting with 'all_gm_v3', and creates a new JSON file that aggregates all 
annotations by unique code names.

For each code name, it collects:
- speaker
- transcript  
- explanation
- score
- score_justification
- file_name (original file name without 'all_gm_v3_')
- conference_name (folder name like '2020NES')
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import re

def find_all_gm_v3_files(base_dir):
    """
    Find all JSON files starting with 'all_gm_v3' in all output folders and subfolders.
    
    Args:
        base_dir: Base directory to search (e.g., 'outputs')
        
    Returns:
        List of tuples (file_path, file_name_without_prefix, conference_name)
    """
    files_found = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist!")
        return files_found
    
    # Recursively search for files starting with 'all_gm_v3'
    for file_path in base_path.rglob("all_gm_ind_detrimental_*.json"):
        # Extract the file name without the 'all_gm_v3_' prefix
        original_name = file_path.name.replace("all_gm_ind_detrimental_", "")
        
        # Extract conference name from the path
        # Look for pattern like outputs/2020NES/... or outputs/2021ABI/...
        path_parts = file_path.parts
        conference_name = None
        
        # Find the conference name in the path
        for part in path_parts:
            if re.match(r'^\d{4}[A-Z]{3}$', part):  # Pattern like 2020NES, 2021ABI
                conference_name = part
                break
        
        if conference_name:
            files_found.append((str(file_path), original_name, conference_name))
        else:
            print(f"Warning: Could not extract conference name from path: {file_path}")
    
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
        
        # Handle the format where annotations has a 'codes' key containing a list
        if isinstance(utterance_annotations, dict) and 'codes' in utterance_annotations:
            codes_list = utterance_annotations['codes']
            if isinstance(codes_list, list):
                for code in codes_list:
                    if isinstance(code, dict):
                        code_name = code.get('code_name', '')
                        if code_name.lower() != 'none':
                            definition = code.get('definition', '')
                            justification = code.get('justification', '')
                            
                            annotations.append({
                                'speaker': speaker,
                                'transcript': transcript,
                                'code_name': code_name,
                                'definition': definition,
                                'justification': justification
                            })
        # Handle the new format where annotations is a dict with code names as keys
        elif isinstance(utterance_annotations, dict):
            # Process each code in the annotations dict
            for code_name, code_data in utterance_annotations.items():
                if code_name.lower() != 'none' and isinstance(code_data, dict):
                    # Extract the annotation details
                    definition = code_data.get('definition', '')
                    justification = code_data.get('justification', '')
                    
                    annotations.append({
                        'speaker': speaker,
                        'transcript': transcript,
                        'code_name': code_name,
                        'definition': definition,
                        'justification': justification
                    })
        # Handle old format if needed
        elif isinstance(utterance_annotations, list):
            for code in utterance_annotations:
                if isinstance(code, dict):
                    code_name = code.get('code_name', '')
                    if code_name.lower() != 'none':
                        definition = code.get('definition', '')
                        justification = code.get('justification', '')
                        
                        annotations.append({
                            'speaker': speaker,
                            'transcript': transcript,
                            'code_name': code_name,
                            'definition': definition,
                            'justification': justification
                        })
    
    return annotations

def aggregate_annotations_by_code(files_list):
    """
    Aggregate annotations by unique code names.
    
    Args:
        files_list: List of tuples (file_path, file_name_without_prefix, conference_name)
        
    Returns:
        Dictionary with code names as keys and lists of occurrences as values
    """
    code_aggregation = defaultdict(list)
    
    for file_path, file_name, conference_name in files_list:
        print(f"Processing: {file_path}")
        
        annotations = extract_annotations_from_file(file_path)
        
        for annotation in annotations:
            code_name = annotation['code_name']
            
            # Add file_name and conference_name to the annotation
            annotation['file_name'] = file_name
            annotation['conference_name'] = conference_name
            
            # Store in aggregation
            code_aggregation[code_name].append({
                'speaker': annotation['speaker'],
                'transcript': annotation['transcript'],
                'justification': annotation['justification'],
                'definition': annotation['definition'],
                # 'score': annotation['score'],
                # 'score_justification': annotation['score_justification'],
                'file_name': annotation['file_name'],
                'conference_name': annotation['conference_name']
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
        # Calculate statistics (no scores in this data format)
        # scores = [occ['score'] for occ in occurrences if isinstance(occ['score'], (int, float))]
        # avg_score = sum(scores) / len(scores) if scores else 0
        
        # Count by conference
        conference_counts = defaultdict(int)
        for occ in occurrences:
            conference_counts[occ['conference_name']] += 1
        
        serializable_data[code_name] = {
            'total_occurrences': len(occurrences),
            # 'average_score': round(avg_score, 2),
            # 'score_distribution': {
            #     '0': len([s for s in scores if s == 0]),
            #     '1': len([s for s in scores if s == 1]),
            #     '2': len([s for s in scores if s == 2]),
            #     '3': len([s for s in scores if s == 3])
            # },
            'conference_distribution': dict(conference_counts),
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
    print("\n" + "="*80)
    print("ANNOTATION AGGREGATION SUMMARY")
    print("="*80)
    
    total_codes = len(aggregated_data)
    total_occurrences = sum(len(occurrences) for occurrences in aggregated_data.values())
    
    print(f"Total unique code names: {total_codes}")
    print(f"Total annotation occurrences: {total_occurrences}")
    
    print(f"\nTop 10 most frequent codes:")
    sorted_codes = sorted(aggregated_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (code_name, occurrences) in enumerate(sorted_codes[:10], 1):
        print(f"{i:2d}. {code_name}: {len(occurrences)} occurrences")
    
    # print(f"\nAll codes and their frequencies:")
    # for code_name, occurrences in sorted(aggregated_data.items()):
    #     scores = [occ['score'] for occ in occurrences if isinstance(occ['score'], (int, float))]
    #     avg_score = sum(scores) / len(scores) if scores else 0
    #     print(f"  {code_name}: {len(occurrences)} occurrences (avg score: {avg_score:.2f})")
    
    # Conference distribution
    print(f"\nConference distribution:")
    conference_totals = defaultdict(int)
    for occurrences in aggregated_data.values():
        for occ in occurrences:
            conference_totals[occ['conference_name']] += 1
    
    for conf, count in sorted(conference_totals.items()):
        print(f"  {conf}: {count} annotations")

def main():
    """
    Main function to run the annotation aggregation.
    """
    # Configuration
    base_dir = "outputs"  # Search all output folders
    output_file = "aggregated_annotations_gemini_detrimental.json"
    
    print("Starting annotation aggregation...")
    print(f"Searching in: {base_dir}")
    
    # Find all relevant files
    files_list = find_all_gm_v3_files(base_dir)
    
    if not files_list:
        print("No files found starting with 'all_gm_ind_detrimental'!")
        return
    
    print(f"Found {len(files_list)} files to process:")
    for file_path, file_name, conference_name in files_list:
        print(f"  - {conference_name}/{file_name}")
    
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