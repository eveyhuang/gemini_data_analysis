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
    features['num_members'] = len(set(all_speakers))
    
    # Get meeting length
    features['meeting_length'] = data.get('total_speaking_length', 0)
    
    # Get all utterances
    utterances = data.get('all_data', [])
    
    # Initialize counters
    annotation_counts = defaultdict(int)
    speaker_annotations = defaultdict(set)
    annotation_scores = defaultdict(list)  # Store scores for mean calculation
    
    # Facilitator-specific counters
    facilitator_utterances = 0
    facilitator_high_quality_utterances = 0
    facilitator_speakers = set()  # Track unique facilitators
    facilitator_scores = []  # Track all scores from facilitator utterances
    
    # Utterance-level quality counters
    negative_utterances = 0
    positive_utterances = 0
    
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
        
        # Check if speaker is facilitator
        speaker = utterance.get('speaker', '')
        role = utterance.get('role', '')
        is_facilitator = 'facilitator' in role.lower()
        
        if is_facilitator:
            facilitator_utterances += 1
            facilitator_speakers.add(speaker)
        
        # Count annotations and collect scores
        annotations = utterance.get('annotations', {})
        
        # Track utterance-level quality
        utterance_has_negative = False
        utterance_has_positive = False
        facilitator_has_high_quality = False
        
        for annotation_type, annotation_data in annotations.items():
            annotation_counts[annotation_type] += 1
            if speaker:
                speaker_annotations[annotation_type].add(speaker)
            
            # Extract score if available
            if isinstance(annotation_data, dict) and 'score' in annotation_data:
                score = annotation_data['score']
                if isinstance(score, (int, float)):
                    annotation_scores[annotation_type].append(score)
                    
                    # Track facilitator scores
                    if is_facilitator:
                        facilitator_scores.append(score)
                    
                    # Track utterance-level quality
                    if score == -1:
                        utterance_has_negative = True
                    if score == 2:
                        utterance_has_positive = True
                        # Check if this is a high-quality score from facilitator
                        if is_facilitator:
                            facilitator_has_high_quality = True
        
        # Count utterance-level quality (only once per utterance)
        if utterance_has_negative:
            negative_utterances += 1
        if utterance_has_positive:
            positive_utterances += 1
        
        # Count high-quality facilitator utterances (only once per utterance)
        if is_facilitator and facilitator_has_high_quality:
            facilitator_high_quality_utterances += 1
    
    # Store basic counts
    features['num_interruption'] = num_interruption
    
    # Calculate percentage of time spent on screenshare
    if features['meeting_length'] > 0:
        features['percent_time_screenshare'] = (time_screenshare / features['meeting_length']) * 100
    else:
        features['percent_time_screenshare'] = 0.0
    
    # Store annotation counts for codebook v4
    features['num_idea_management'] = annotation_counts.get('Idea Management', 0)
    features['num_information_seeking'] = annotation_counts.get('Information Seeking', 0)
    features['num_knowledge_sharing'] = annotation_counts.get('Knowledge Sharing', 0)
    features['num_evaluation_practices'] = annotation_counts.get('Evaluation Practices', 0)
    features['num_relational_climate'] = annotation_counts.get('Relational Climate', 0)
    features['num_participation_dynamics'] = annotation_counts.get('Participation Dynamics', 0)
    features['num_coordination_decision'] = annotation_counts.get('Coordination and Decision Practices', 0)
    features['num_integration_practices'] = annotation_counts.get('Integration Practices', 0)
    
    # Count unique speakers for each annotation type (codebook v4)
    features['num_people_idea_management'] = len(speaker_annotations.get('Idea Management', set()))
    features['num_people_information_seeking'] = len(speaker_annotations.get('Information Seeking', set()))
    features['num_people_knowledge_sharing'] = len(speaker_annotations.get('Knowledge Sharing', set()))
    features['num_people_evaluation_practices'] = len(speaker_annotations.get('Evaluation Practices', set()))
    features['num_people_relational_climate'] = len(speaker_annotations.get('Relational Climate', set()))
    features['num_people_participation_dynamics'] = len(speaker_annotations.get('Participation Dynamics', set()))
    features['num_people_coordination_decision'] = len(speaker_annotations.get('Coordination and Decision Practices', set()))
    features['num_people_integration_practices'] = len(speaker_annotations.get('Integration Practices', set()))
    
    # Calculate mean scores for each code (codebook v4)
    features['mean_score_idea_management'] = np.mean(annotation_scores.get('Idea Management', [])) if annotation_scores.get('Idea Management') else 0.0
    features['mean_score_information_seeking'] = np.mean(annotation_scores.get('Information Seeking', [])) if annotation_scores.get('Information Seeking') else 0.0
    features['mean_score_knowledge_sharing'] = np.mean(annotation_scores.get('Knowledge Sharing', [])) if annotation_scores.get('Knowledge Sharing') else 0.0
    features['mean_score_evaluation_practices'] = np.mean(annotation_scores.get('Evaluation Practices', [])) if annotation_scores.get('Evaluation Practices') else 0.0
    features['mean_score_relational_climate'] = np.mean(annotation_scores.get('Relational Climate', [])) if annotation_scores.get('Relational Climate') else 0.0
    features['mean_score_participation_dynamics'] = np.mean(annotation_scores.get('Participation Dynamics', [])) if annotation_scores.get('Participation Dynamics') else 0.0
    features['mean_score_coordination_decision'] = np.mean(annotation_scores.get('Coordination and Decision Practices', [])) if annotation_scores.get('Coordination and Decision Practices') else 0.0
    features['mean_score_integration_practices'] = np.mean(annotation_scores.get('Integration Practices', [])) if annotation_scores.get('Integration Practices') else 0.0
    
    # Calculate overall mean score across all codes
    all_scores = []
    for code_scores in annotation_scores.values():
        all_scores.extend(code_scores)
    features['mean_score_overall'] = np.mean(all_scores) if all_scores else 0.0
    
    # Calculate total score (sum of all scores)
    features['total_score'] = sum(all_scores) if all_scores else 0.0
    
    # Calculate total utterances
    total_utterances = len(utterances)
    features['total_utterances'] = total_utterances
    
    # Calculate negative utterance ratio (proportion of utterances with -1 scores)
    features['negative_utterance_ratio'] = negative_utterances / total_utterances if total_utterances > 0 else 0.0
    
    # Calculate positive intensity (proportion of utterances with score 2)
    features['positive_intensity'] = positive_utterances / total_utterances if total_utterances > 0 else 0.0
    
    # Calculate facilitator-specific features
    features['num_facilitator'] = len(facilitator_speakers)
    features['facilitator_dominance_ratio'] = facilitator_utterances / total_utterances if total_utterances > 0 else 0.0
    features['facilitator_high_quality_ratio'] = facilitator_high_quality_utterances / facilitator_utterances if facilitator_utterances > 0 else 0.0
    features['facilitator_average_score'] = np.mean(facilitator_scores) if facilitator_scores else 0.0
    
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