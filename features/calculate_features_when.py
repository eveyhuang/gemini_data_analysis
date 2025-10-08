import json
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_features_by_when(data, conf_name=None, session_name=None):
    """Calculate features for each time segment (beginning, middle, end) separately."""
    # Get all speakers
    all_speakers = data.get('all_speakers', [])
    num_members = len(set(all_speakers))
    
    # Get meeting length
    meeting_length = data.get('total_speaking_length', 0)
    
    # Get all utterances
    utterances = data.get('all_data', [])
    
    # Split utterances by when field
    beginning_utterances = []
    middle_utterances = []
    end_utterances = []
    
    for utterance in utterances:
        # Get the 'when' field directly from the utterance
        when_value = utterance.get('when', '')
        
        # Categorize utterance based on when value
        if when_value == 'beginning':
            beginning_utterances.append(utterance)
        elif when_value == 'middle':
            middle_utterances.append(utterance)
        elif when_value == 'end':
            end_utterances.append(utterance)
    
    # Calculate features for each time segment separately
    beginning_features = calculate_segment_features(beginning_utterances, 'beginning', num_members, meeting_length, conf_name, session_name)
    middle_features = calculate_segment_features(middle_utterances, 'middle', num_members, meeting_length, conf_name, session_name)
    end_features = calculate_segment_features(end_utterances, 'end', num_members, meeting_length, conf_name, session_name)
    
    return {
        'beginning': beginning_features,
        'middle': middle_features,
        'end': end_features
    }

def calculate_segment_features(utterances, segment_name, num_members, meeting_length, conf_name=None, session_name=None):
    """Calculate features for a specific time segment."""
    features = {}
    
    # Add basic session info
    features['num_members'] = num_members
    features['meeting_length'] = meeting_length
    features['segment'] = segment_name
    features['conf_name'] = conf_name
    features['session'] = session_name
    
    if not utterances:
        # Return zero features if no utterances in this segment
        return get_zero_features(segment_name, num_members, meeting_length, conf_name, session_name)
    
    # Initialize counters
    annotation_counts = defaultdict(int)
    speaker_annotations = defaultdict(set)
    annotation_scores = defaultdict(list)
    
    # Facilitator-specific counters
    facilitator_utterances = 0
    facilitator_high_quality_utterances = 0
    facilitator_speakers = set()
    facilitator_scores = []
    
    # Utterance-level quality counters
    negative_utterances = 0
    positive_utterances = 0
    
    # Count interruptions and screenshare
    num_interruption = 0
    time_screenshare = 0
    
    # Track unique speakers in this segment
    segment_speakers = set()
    
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
        
        # Track unique speakers in this segment
        if speaker:
            segment_speakers.add(speaker)
        
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
    
    # Calculate percentage of time spent on screenshare for this segment
    segment_length = sum(float(u.get('speaking_duration', 0)) for u in utterances)
    if segment_length > 0:
        features['percent_time_screenshare'] = (time_screenshare / segment_length) * 100
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
    
    # Count unique speakers for each annotation type
    features['num_people_idea_management'] = len(speaker_annotations.get('Idea Management', set()))
    features['num_people_information_seeking'] = len(speaker_annotations.get('Information Seeking', set()))
    features['num_people_knowledge_sharing'] = len(speaker_annotations.get('Knowledge Sharing', set()))
    features['num_people_evaluation_practices'] = len(speaker_annotations.get('Evaluation Practices', set()))
    features['num_people_relational_climate'] = len(speaker_annotations.get('Relational Climate', set()))
    features['num_people_participation_dynamics'] = len(speaker_annotations.get('Participation Dynamics', set()))
    features['num_people_coordination_decision'] = len(speaker_annotations.get('Coordination and Decision Practices', set()))
    features['num_people_integration_practices'] = len(speaker_annotations.get('Integration Practices', set()))
    
    # Calculate mean scores for each code
    features['mean_score_idea_management'] = np.mean(annotation_scores.get('Idea Management', [])) if annotation_scores.get('Idea Management') else 0.0
    features['mean_score_information_seeking'] = np.mean(annotation_scores.get('Information Seeking', [])) if annotation_scores.get('Information Seeking') else 0.0
    features['mean_score_knowledge_sharing'] = np.mean(annotation_scores.get('Knowledge Sharing', [])) if annotation_scores.get('Knowledge Sharing') else 0.0
    features['mean_score_evaluation_practices'] = np.mean(annotation_scores.get('Evaluation Practices', [])) if annotation_scores.get('Evaluation Practices') else 0.0
    features['mean_score_relational_climate'] = np.mean(annotation_scores.get('Relational Climate', [])) if annotation_scores.get('Relational Climate') else 0.0
    features['mean_score_participation_dynamics'] = np.mean(annotation_scores.get('Participation Dynamics', [])) if annotation_scores.get('Participation Dynamics') else 0.0
    features['mean_score_coordination_decision'] = np.mean(annotation_scores.get('Coordination and Decision Practices', [])) if annotation_scores.get('Coordination and Decision Practices') else 0.0
    features['mean_score_integration_practices'] = np.mean(annotation_scores.get('Integration Practices', [])) if annotation_scores.get('Integration Practices') else 0.0
    
    # Calculate overall mean score across all codes for this segment
    all_scores = []
    for code_scores in annotation_scores.values():
        all_scores.extend(code_scores)
    features['mean_score_overall'] = np.mean(all_scores) if all_scores else 0.0
    
    # Calculate total score for this segment
    features['total_score'] = sum(all_scores) if all_scores else 0.0
    
    # Calculate total utterances for this segment
    total_utterances = len(utterances)
    features['total_utterances'] = total_utterances
    
    # Calculate negative utterance ratio for this segment
    features['negative_utterance_ratio'] = negative_utterances / total_utterances if total_utterances > 0 else 0.0
    
    # Calculate positive intensity for this segment
    features['positive_intensity'] = positive_utterances / total_utterances if total_utterances > 0 else 0.0
    
    # Calculate facilitator-specific features for this segment
    features['num_facilitator'] = len(facilitator_speakers)
    features['facilitator_dominance_ratio'] = facilitator_utterances / total_utterances if total_utterances > 0 else 0.0
    features['facilitator_high_quality_ratio'] = facilitator_high_quality_utterances / facilitator_utterances if facilitator_utterances > 0 else 0.0
    features['facilitator_average_score'] = np.mean(facilitator_scores) if facilitator_scores else 0.0
    
    # Add number of unique speakers in this segment
    features['num_segment_members'] = len(segment_speakers)
    
    # Calculate segment length (time span from first to last utterance in seconds)
    if utterances:
        # Parse start and end times to calculate actual time span
        start_times = []
        end_times = []
        
        for utterance in utterances:
            start_time = utterance.get('start_time', '00:00')
            end_time = utterance.get('end_time', '00:00')
            
            # Convert time strings to seconds
            def time_to_seconds(time_str):
                parts = time_str.split(':')
                if len(parts) == 2:  # MM:SS format
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:  # HH:MM:SS format
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return 0
            
            start_times.append(time_to_seconds(start_time))
            end_times.append(time_to_seconds(end_time))
        
        # Calculate segment length as time span from earliest start to latest end
        if start_times and end_times:
            segment_start = min(start_times)
            segment_end = max(end_times)
            segment_length = segment_end - segment_start
        else:
            segment_length = 0
    else:
        segment_length = 0
    
    features['segment_length'] = segment_length
    
    return features

def get_zero_features(segment_name, num_members, meeting_length, conf_name=None, session_name=None):
    """Return zero values for all features when a segment has no utterances."""
    features = {}
    
    # Add basic session info
    features['num_members'] = num_members
    features['meeting_length'] = meeting_length
    features['segment'] = segment_name
    features['conf_name'] = conf_name
    features['session'] = session_name
    
    # Basic counts
    features['num_interruption'] = 0
    features['percent_time_screenshare'] = 0.0
    
    # Annotation counts
    annotation_types = [
        'idea_management', 'information_seeking', 'knowledge_sharing', 
        'evaluation_practices', 'relational_climate', 'participation_dynamics',
        'coordination_decision', 'integration_practices'
    ]
    
    for annotation_type in annotation_types:
        features[f'num_{annotation_type}'] = 0
        features[f'num_people_{annotation_type}'] = 0
        features[f'mean_score_{annotation_type}'] = 0.0
    
    # Overall scores
    features['mean_score_overall'] = 0.0
    features['total_score'] = 0.0
    features['total_utterances'] = 0
    features['negative_utterance_ratio'] = 0.0
    features['positive_intensity'] = 0.0
    
    # Facilitator features
    features['num_facilitator'] = 0
    features['facilitator_dominance_ratio'] = 0.0
    features['facilitator_high_quality_ratio'] = 0.0
    features['facilitator_average_score'] = 0.0
    
    # Number of unique speakers in this segment
    features['num_segment_members'] = 0
    
    # Segment length (zero for empty segments)
    features['segment_length'] = 0.0
    
    return features

def process_session_file(file_path):
    """Process a single session file and calculate all features by time segments."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract conf_name and session from file path
    # Expected format: data/{conf_name}/session_data/{session}.json
    path_parts = file_path.split('/')
    conf_name = None
    session_name = None
    
    if 'data' in path_parts:
        data_idx = path_parts.index('data')
        if data_idx + 1 < len(path_parts):
            conf_name = path_parts[data_idx + 1]
    
    # Extract session name from filename (remove .json extension)
    filename = path_parts[-1] if path_parts else ''
    session_name = filename.replace('.json', '') if filename.endswith('.json') else filename
    
    features = calculate_features_by_when(data, conf_name, session_name)
    return features

def main():
    parser = argparse.ArgumentParser(description="Featurize session JSON files with time segment analysis.")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., 2020NES)")
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = f'data/{dataset}/featurized_with_when'
    os.makedirs(output_dir, exist_ok=True)
    
    data_dir = f'data/{dataset}/session_data'
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and 'person_to_team' not in filename and 'outcome' not in filename:
            input_path = os.path.join(data_dir, filename)
            
            print(f"Processing {filename}...")
            segment_features = process_session_file(input_path)
            
            # Save three separate files for each time segment
            for segment_name, features in segment_features.items():
                # Create filename with segment suffix
                base_name = filename.replace('.json', '')
                output_filename = f'features_{base_name}_{segment_name}.json'
                output_path = os.path.join(output_dir, output_filename)
                
                # Save features to JSON file
                with open(output_path, 'w') as f:
                    json.dump(features, f, indent=2)
                
                print(f"  Saved {segment_name} features to {output_filename}")

if __name__ == "__main__":
    main()
