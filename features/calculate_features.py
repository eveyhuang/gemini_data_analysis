import json
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_features(data):
    """Calculate all features specified in the feature guide."""
    features = {}
    
    # Get all speakers
    all_speakers = data.get('all_speakers', [])
    features['num_members'] = len(all_speakers)
    
    # Get meeting length
    features['meeting_length'] = data.get('total_speaking_length', 0)
    
    # Get all utterances
    utterances = data.get('all_data', [])
    
    # Initialize counters
    annotation_counts = defaultdict(int)
    speaker_annotations = defaultdict(set)
    
    # Count interruptions
    num_interruption = 0
    time_screenshare = 0
    
    for utterance in utterances:
        # Count interruptions
        if utterance.get('interruption') == 'Yes':
            num_interruption += 1
        
        # Calculate screenshare time
        if utterance.get('screenshare') == 'Yes':
            time_screenshare += float(utterance.get('speaking_duration', 0))
        
        # Count annotations
        annotations = utterance.get('annotations', {})
        speaker = utterance.get('speaker', '')
        
        for annotation_type in annotations:
            annotation_counts[annotation_type] += 1
            if speaker:
                speaker_annotations[annotation_type].add(speaker)
    
    # Store basic counts
    features['num_interruption'] = num_interruption
    
    # Calculate percentage of time spent on screenshare
    if features['meeting_length'] > 0:
        features['percent_time_screenshare'] = (time_screenshare / features['meeting_length']) * 100
    else:
        features['percent_time_screenshare'] = 0.0
    
    # Store annotation counts
    features['num_propose_new_idea'] = annotation_counts.get('propose new idea', 0)
    features['num_develop_idea'] = annotation_counts.get('develop idea', 0)
    features['num_ask_question'] = annotation_counts.get('ask question', 0)
    features['num_signal_expertise'] = annotation_counts.get('signal expertise', 0)
    features['num_identify_gap'] = annotation_counts.get('identify gap', 0)
    features['num_acknowledge_contribution'] = annotation_counts.get('acknowledge contribution', 0)
    features['num_supportive_response'] = annotation_counts.get('supportive response', 0)
    features['num_critical_response'] = annotation_counts.get('critical response', 0)
    features['num_offer_feedback'] = annotation_counts.get('offer feedback', 0)
    features['num_summarize_conversation'] = annotation_counts.get('summarize conversation', 0)
    features['num_express_humor'] = annotation_counts.get('express humor', 0)
    features['num_encourage_participation'] = annotation_counts.get('encourage participation', 0)
    features['num_process_management'] = annotation_counts.get('process management', 0)
    features['num_assign_task'] = annotation_counts.get('assign task', 0)
    features['num_clarify_goal'] = annotation_counts.get('clarify goal', 0)
    features['num_confirm_decision'] = annotation_counts.get('confirm decision', 0)
    
    # Count unique speakers for each annotation type
    features['num_people_ask_question'] = len(speaker_annotations.get('ask question', set()))
    features['num_people_identify_gap'] = len(speaker_annotations.get('identify gap', set()))
    features['num_people_supportive_response'] = len(speaker_annotations.get('supportive response', set()))
    features['num_people_critical_response'] = len(speaker_annotations.get('critical response', set()))
    features['num_people_offer_feedback'] = len(speaker_annotations.get('offer feedback', set()))
    features['num_people_summarize_conversation'] = len(speaker_annotations.get('summarize conversation', set()))
    features['num_people_express_humor'] = len(speaker_annotations.get('express humor', set()))
    features['num_people_encourage_participation'] = len(speaker_annotations.get('encourage participation', set()))
    features['num_people_process_management'] = len(speaker_annotations.get('process management', set()))
    features['num_people_assign_task'] = len(speaker_annotations.get('assign task', set()))
    features['num_people_clarify_goal'] = len(speaker_annotations.get('clarify goal', set()))
    features['num_people_confirm_decision'] = len(speaker_annotations.get('confirm decision', set()))
    
    return features

def process_session_file(file_path):
    """Process a single session file and calculate all features."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    features = calculate_features(data)
    return features

def main():
    parser = argparse.ArgumentParser(description="Featurize session JSON files for a given dataset.")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., 2020NES)")
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = f'data/{dataset}/featurized data'
    os.makedirs(output_dir, exist_ok=True)
    
    data_dir = f'data/{dataset}/session_data'
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and 'person_to_team' not in filename and 'outcome' not in filename:
            input_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, f'features_{filename}')
            
            print(f"Processing {filename}...")
            features = process_session_file(input_path)
            
            # Save features to JSON file
            with open(output_path, 'w') as f:
                json.dump(features, f, indent=2)
            
            print(f"Saved features to {output_path}")

if __name__ == "__main__":
    main()