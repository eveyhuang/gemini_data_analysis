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
    features['count_members'] = num_members
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
    
    # Track high/low quality codes per code type
    has_high_quality = defaultdict(bool)  # Track if any score=2 for each code
    has_low_quality = defaultdict(bool)   # Track if any score=-1 for each code
    
    # Facilitator-specific counters
    facilitator_utterances = 0
    facilitator_high_quality_utterances = 0
    facilitator_speakers = set()
    facilitator_scores = []
    
    # Utterance-level quality counters
    negative_utterances = 0
    positive_utterances = 0
    
    # Count interruptions and screenshare
    count_interruption = 0
    time_screenshare = 0
    
    # Track unique speakers in this segment
    segment_speakers = set()
    
    for utterance in utterances:
        # Count interruptions
        if utterance.get('interruption') == 'Yes':
            count_interruption += 1
        
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
                    
                    # Track high/low quality per code type
                    if score == 2:
                        has_high_quality[annotation_type] = True
                    if score == -1:
                        has_low_quality[annotation_type] = True
                    
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
    features['count_interruption'] = count_interruption
    
    # Calculate percentage of time spent on screenshare for this segment
    segment_length = sum(float(u.get('speaking_duration', 0)) for u in utterances)
    if segment_length > 0:
        features['percent_time_screenshare'] = (time_screenshare / segment_length) * 100
    else:
        features['percent_time_screenshare'] = 0.0
    
    # Store annotation counts for codebook v4
    features['count_idea_management'] = annotation_counts.get('Idea Management', 0)
    features['count_information_seeking'] = annotation_counts.get('Information Seeking', 0)
    features['count_knowledge_sharing'] = annotation_counts.get('Knowledge Sharing', 0)
    features['count_evaluation_practices'] = annotation_counts.get('Evaluation Practices', 0)
    features['count_relational_climate'] = annotation_counts.get('Relational Climate', 0)
    features['count_participation_dynamics'] = annotation_counts.get('Participation Dynamics', 0)
    features['count_coordination_decision'] = annotation_counts.get('Coordination and Decision Practices', 0)
    features['count_integration_practices'] = annotation_counts.get('Integration Practices', 0)
    
    # Count unique speakers for each annotation type
    features['count_people_idea_management'] = len(speaker_annotations.get('Idea Management', set()))
    features['count_people_information_seeking'] = len(speaker_annotations.get('Information Seeking', set()))
    features['count_people_knowledge_sharing'] = len(speaker_annotations.get('Knowledge Sharing', set()))
    features['count_people_evaluation_practices'] = len(speaker_annotations.get('Evaluation Practices', set()))
    features['count_people_relational_climate'] = len(speaker_annotations.get('Relational Climate', set()))
    features['count_people_participation_dynamics'] = len(speaker_annotations.get('Participation Dynamics', set()))
    features['count_people_coordination_decision'] = len(speaker_annotations.get('Coordination and Decision Practices', set()))
    features['count_people_integration_practices'] = len(speaker_annotations.get('Integration Practices', set()))
    
    # Binary features for high quality (score=2) per code
    features['has_high_quality_idea_management'] = 1 if has_high_quality.get('Idea Management', False) else 0
    features['has_high_quality_information_seeking'] = 1 if has_high_quality.get('Information Seeking', False) else 0
    features['has_high_quality_knowledge_sharing'] = 1 if has_high_quality.get('Knowledge Sharing', False) else 0
    features['has_high_quality_evaluation_practices'] = 1 if has_high_quality.get('Evaluation Practices', False) else 0
    features['has_high_quality_relational_climate'] = 1 if has_high_quality.get('Relational Climate', False) else 0
    features['has_high_quality_participation_dynamics'] = 1 if has_high_quality.get('Participation Dynamics', False) else 0
    features['has_high_quality_coordination_decision'] = 1 if has_high_quality.get('Coordination and Decision Practices', False) else 0
    features['has_high_quality_integration_practices'] = 1 if has_high_quality.get('Integration Practices', False) else 0
    
    # Binary features for low quality (score=-1) per code
    features['has_low_quality_idea_management'] = 1 if has_low_quality.get('Idea Management', False) else 0
    features['has_low_quality_information_seeking'] = 1 if has_low_quality.get('Information Seeking', False) else 0
    features['has_low_quality_knowledge_sharing'] = 1 if has_low_quality.get('Knowledge Sharing', False) else 0
    features['has_low_quality_evaluation_practices'] = 1 if has_low_quality.get('Evaluation Practices', False) else 0
    features['has_low_quality_relational_climate'] = 1 if has_low_quality.get('Relational Climate', False) else 0
    features['has_low_quality_participation_dynamics'] = 1 if has_low_quality.get('Participation Dynamics', False) else 0
    features['has_low_quality_coordination_decision'] = 1 if has_low_quality.get('Coordination and Decision Practices', False) else 0
    features['has_low_quality_integration_practices'] = 1 if has_low_quality.get('Integration Practices', False) else 0
    
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
    
    # Calculate low quality ratio for this segment (proportion of utterances with -1 scores)
    features['low_quality_ratio'] = negative_utterances / total_utterances if total_utterances > 0 else 0.0
    
    # Calculate high quality ratio for this segment (proportion of utterances with score 2)
    features['high_quality_ratio'] = positive_utterances / total_utterances if total_utterances > 0 else 0.0
    
    # Calculate facilitator-specific features for this segment
    features['count_facilitator'] = len(facilitator_speakers)
    features['facilitator_dominance_ratio'] = facilitator_utterances / total_utterances if total_utterances > 0 else 0.0
    features['facilitator_high_quality_ratio'] = facilitator_high_quality_utterances / facilitator_utterances if facilitator_utterances > 0 else 0.0
    features['facilitator_average_score'] = np.mean(facilitator_scores) if facilitator_scores else 0.0
    
    # Add number of unique speakers in this segment
    features['count_segment_members'] = len(segment_speakers)
    
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
    features['count_members'] = num_members
    features['meeting_length'] = meeting_length
    features['segment'] = segment_name
    features['conf_name'] = conf_name
    features['session'] = session_name
    
    # Basic counts
    features['count_interruption'] = 0
    features['percent_time_screenshare'] = 0.0
    
    # Annotation counts
    annotation_types = [
        'idea_management', 'information_seeking', 'knowledge_sharing', 
        'evaluation_practices', 'relational_climate', 'participation_dynamics',
        'coordination_decision', 'integration_practices'
    ]
    
    for annotation_type in annotation_types:
        features[f'count_{annotation_type}'] = 0
        features[f'count_people_{annotation_type}'] = 0
        features[f'mean_score_{annotation_type}'] = 0.0
        features[f'has_high_quality_{annotation_type}'] = 0
        features[f'has_low_quality_{annotation_type}'] = 0
    
    # Overall scores
    features['mean_score_overall'] = 0.0
    features['total_score'] = 0.0
    features['total_utterances'] = 0
    features['low_quality_ratio'] = 0.0
    features['high_quality_ratio'] = 0.0
    
    # Facilitator features
    features['count_facilitator'] = 0
    features['facilitator_dominance_ratio'] = 0.0
    features['facilitator_high_quality_ratio'] = 0.0
    features['facilitator_average_score'] = 0.0
    
    # Number of unique speakers in this segment
    features['count_segment_members'] = 0
    
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

def process_dataset(dataset):
    """Process a single dataset folder with time segment analysis."""
    output_dir = f'data/{dataset}/featurized_with_when'
    data_dir = f'data/{dataset}/session_data'
    
    # Check if session_data folder exists
    if not os.path.exists(data_dir):
        print(f"  ⚠️  Skipping {dataset}: no 'session_data' folder found")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and 'person_to_team' not in filename and 'outcome' not in filename:
            input_path = os.path.join(data_dir, filename)
            
            print(f"  Processing {filename}...")
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
                
            processed_count += 1
    
    print(f"  ✅ Saved {processed_count * 3} feature files (3 segments each) to {output_dir}")
    return processed_count


def main():
    parser = argparse.ArgumentParser(description="Featurize session JSON files with time segment analysis.")
    parser.add_argument("dataset", type=str, nargs='?', default=None,
                        help="Dataset name (e.g., 2020NES). If not provided, processes all datasets in /data")
    args = parser.parse_args()

    if args.dataset:
        # Process single dataset
        print(f"📂 Processing dataset: {args.dataset}")
        process_dataset(args.dataset)
    else:
        # Process all datasets in /data folder
        data_root = 'data'
        if not os.path.exists(data_root):
            print(f"❌ Error: '{data_root}' directory not found")
            return
        
        # Get all subdirectories in /data
        datasets = [d for d in os.listdir(data_root) 
                    if os.path.isdir(os.path.join(data_root, d))]
        
        if not datasets:
            print(f"❌ No dataset folders found in '{data_root}'")
            return
        
        print(f"🔍 Found {len(datasets)} dataset(s) in '{data_root}': {', '.join(datasets)}")
        print("=" * 60)
        
        total_processed = 0
        for dataset in sorted(datasets):
            print(f"\n📂 Processing dataset: {dataset}")
            total_processed += process_dataset(dataset)
        
        print("\n" + "=" * 60)
        print(f"✅ Done! Processed {total_processed} total session files across {len(datasets)} dataset(s)")
        print(f"   (Each session creates 3 segment files: beginning, middle, end)")


if __name__ == "__main__":
    main()
