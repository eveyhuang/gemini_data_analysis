import json
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from scipy import stats
import re
import argparse

def calculate_core_verbal_features(utterances):
    """Calculate core verbal team behavior features."""
    features = defaultdict(int)
    
    # Define feature mappings (annotation name -> feature name)
    feature_mappings = {
        'present new idea': 'num_present_new_idea',
        'expand on existing idea': 'num_expand_on_existing_idea',
        'provide supporting evidence': 'num_provide_supporting_evidence',
        'explain or define term or concept': 'num_explain_define_term',
        'propose decision': 'num_propose_decision',
        'confirm decision': 'num_confirm_decision',
        'express alternative decision': 'num_express_alternative_decision',
        'offer constructive criticism': 'num_offer_constructive_criticism',
        'reject idea': 'num_reject_idea',
        'resolve conflict': 'num_resolve_conflict',
        'assign task': 'num_assign_task',
        'encourage participation': 'num_encourage_participation',
        'acknowledge contribution': 'num_acknowledge_contribution',
        'express enthusiasm': 'num_express_enthusiasm',
        'express frustration': 'num_express_frustration',
        'express humor': 'num_express_humor',
        'express agreement': 'num_express_agreement'
    }
    
    for utterance in utterances:
        # Count basic features from annotations
        annotations = utterance.get('annotations', {})
        for annotation, feature_name in feature_mappings.items():
            if annotation in annotations:
                features[feature_name] += 1
    
    # Calculate ratios
    if features['num_present_new_idea'] > 0:
        features['elaboration_to_idea_ratio'] = features['num_expand_on_existing_idea'] / features['num_present_new_idea']
    
    if features['num_propose_decision'] > 0:
        features['decision_closure_ratio'] = features['num_confirm_decision'] / features['num_propose_decision']
    
    if features['num_express_agreement'] > 0:
        features['criticism_to_agreement_ratio'] = features['num_offer_constructive_criticism'] / features['num_express_agreement']
    
    # Calculate net scores
    features['net_positive_conflict_score'] = features['num_resolve_conflict'] - features['num_reject_idea']
    
    # Calculate leadership action count
    features['leadership_action_count'] = (features['num_assign_task'] + features['num_propose_decision'] +
                                         features['num_confirm_decision'] + features['num_encourage_participation'])
    
    # Calculate engagement positivity score
    features['engagement_positivity_score'] = (features['num_acknowledge_contribution'] + features['num_express_enthusiasm'] +
                                             features['num_express_humor'] - features['num_express_frustration'])
    
    return features

def calculate_speaker_engagement_features(utterances):
    """Calculate speaker engagement and dominance features."""
    features = {}
    
    # Calculate speaking durations
    speaking_durations = []
    speaker_durations = defaultdict(float)
    
    for utterance in utterances:
        duration = float(utterance.get('speaking_duration', 0))
        speaking_durations.append(duration)
        speaker = utterance.get('speaker', '')
        if speaker:
            speaker_durations[speaker] += duration
    
    if speaking_durations:
        speaking_durations = np.array(speaking_durations)
        features['avg_speaking_duration'] = np.mean(speaking_durations)
        features['speaking_variance'] = np.std(speaking_durations)
        
        # Calculate speaking time ratio per speaker
        total_time = np.sum(speaking_durations)
        if total_time > 0:
            features['speaking_time_ratio'] = {
                speaker: duration / total_time 
                for speaker, duration in speaker_durations.items()
            }
        
        # Calculate participation entropy
        if len(speaking_durations) > 0:
            probs = speaking_durations / total_time
            features['participation_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
    
    return features

def extract_numeric_value(value):
    """Extract numeric value from string that may contain confidence levels."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        # Remove everything after and including '['
        value = value.split('[')[0].strip()
        try:
            return float(value)
        except ValueError:
            return 0
    return 0

def calculate_nonverbal_features(utterances):
    """Calculate nonverbal features."""
    features = {}
    
    total_nods = 0
    total_gestures = 0
    total_eye_contact = 0
    total_utterances = len(utterances)
    
    for utterance in utterances:
        # Extract numeric values, handling confidence levels
        nods = extract_numeric_value(utterance.get('nods_others', 0))
        gestures = extract_numeric_value(utterance.get('gestures', 0))
        eye_contact = extract_numeric_value(utterance.get('eye_contact', 0))
        
        total_nods += nods
        total_gestures += gestures
        total_eye_contact += eye_contact
    
    if total_utterances > 0:
        features['avg_nods_per_utterance'] = total_nods / total_utterances
        features['avg_gestures_per_utterance'] = total_gestures / total_utterances
        features['avg_eye_contact_per_utterance'] = total_eye_contact / total_utterances
    
    return features

def calculate_cognitive_load_features(utterances):
    """Calculate cognitive load features."""
    features = {}
    
    total_distracted = 0
    total_confused = 0
    total_engaged = 0
    total_utterances = len(utterances)
    
    for utterance in utterances:
        # Extract numeric values, handling confidence levels
        distracted = extract_numeric_value(utterance.get('distracted_others', 0))
        confused = extract_numeric_value(utterance.get('confused_others', 0))
        engaged = extract_numeric_value(utterance.get('engaged_others', 0))
        
        total_distracted += distracted
        total_confused += confused
        total_engaged += engaged
    
    if total_utterances > 0:
        features['avg_distraction_per_utterance'] = total_distracted / total_utterances
        features['avg_confusion_per_utterance'] = total_confused / total_utterances
        features['avg_engagement_per_utterance'] = total_engaged / total_utterances
    
    return features

def calculate_gesture_features(utterances):
    """Calculate gesture and visual communication features."""
    features = {}
    
    gesture_types = defaultdict(int)
    speaker_gestures = defaultdict(set)
    total_gestures = 0
    
    for utterance in utterances:
        gesture = utterance.get('hand_gesture', '')
        speaker = utterance.get('speaker', '')
        
        if gesture and gesture != 'None':  # Only count non-empty gestures
            gesture_types[gesture] += 1
            if speaker:
                speaker_gestures[speaker].add(gesture)
            total_gestures += 1
    
    if gesture_types:
        features['gesture_type_distribution'] = dict(gesture_types)
        features['gesture_variety'] = {speaker: len(gestures) for speaker, gestures in speaker_gestures.items()}
        
        if utterances:
            features['gesture_count_ratio'] = total_gestures / len(utterances)
    
    return features

def calculate_turn_taking_features(utterances):
    """Calculate turn-taking and conversational fluidity features."""
    features = {}
    
    interruptions = defaultdict(int)
    overlaps = defaultdict(int)
    interruption_received = defaultdict(int)
    
    for utterance in utterances:
        speaker = utterance.get('speaker', '')
        if speaker:
            if utterance.get('interruption', False) == 'Yes':  # Fixed to check for 'Yes' string
                interruptions[speaker] += 1
            if utterance.get('overlap', False) == 'Yes':  # Fixed to check for 'Yes' string
                overlaps[speaker] += 1
            if utterance.get('interrupted', False) == 'Yes':  # Fixed to check for 'Yes' string
                interruption_received[speaker] += 1
    
    if interruptions or overlaps or interruption_received:
        features['num_interruptions'] = sum(interruptions.values())
        features['num_overlaps'] = sum(overlaps.values())
        features['interruption_rate_by_speaker'] = dict(interruptions)
        features['interruption_received'] = dict(interruption_received)
    
    return features

def calculate_screensharing_features(utterances):
    """Calculate screensharing and coordination features."""
    features = {}
    
    screenshare_segments = []
    current_segment = {'start': None, 'duration': 0}
    screenshare_content = []
    
    for utterance in utterances:
        if utterance.get('screenshare', False) == 'Yes':  # Fixed to check for 'Yes' string
            if current_segment['start'] is None:
                current_segment['start'] = utterance.get('timestamp', 0)
            current_segment['duration'] += float(utterance.get('speaking_duration', 0))
            
            if utterance.get('screenshare_content') and utterance['screenshare_content'] != 'None':
                screenshare_content.append(utterance['screenshare_content'])
        elif current_segment['start'] is not None:
            screenshare_segments.append(current_segment)
            current_segment = {'start': None, 'duration': 0}
    
    if current_segment['start'] is not None:
        screenshare_segments.append(current_segment)
    
    if screenshare_segments:
        features['screenshare_count'] = len(screenshare_segments)
        features['screenshare_duration'] = sum(segment['duration'] for segment in screenshare_segments)
        features['avg_screenshare_segment_length'] = features['screenshare_duration'] / len(screenshare_segments)
    
    if screenshare_content:
        features['screenshare_content'] = screenshare_content
    
    # Calculate screenshare decision overlap
    screenshare_decisions = 0
    total_screenshare = 0
    for utterance in utterances:
        if utterance.get('screenshare', False) == 'Yes':  # Fixed to check for 'Yes' string
            total_screenshare += 1
            annotations = utterance.get('annotations', {})
            if 'propose decision' in annotations or 'assign task' in annotations:
                screenshare_decisions += 1
    
    if total_screenshare > 0:
        features['screenshare_decision_overlap'] = screenshare_decisions / total_screenshare
    
    return features

def calculate_temporal_features(utterances):
    """Calculate temporal and sequential metrics using actual speaking durations."""
    features = {}
    
    # Calculate total duration of the session
    total_duration = sum(float(u.get('speaking_duration', 0)) for u in utterances)
    mid_point_duration = total_duration / 2
    
    # Split utterances into first and second half based on duration
    first_half_duration = 0
    first_half_ideas = 0
    second_half_ideas = 0
    
    for utterance in utterances:
        duration = float(utterance.get('speaking_duration', 0))
        if first_half_duration < mid_point_duration:
            first_half_duration += duration
            if 'present new idea' in utterance.get('annotations', {}):
                first_half_ideas += 1
        else:
            if 'present new idea' in utterance.get('annotations', {}):
                second_half_ideas += 1
    
    features['ideas_first_half'] = first_half_ideas
    features['ideas_second_half'] = second_half_ideas
    
    # Calculate decision lag in seconds
    decision_lags = []
    current_time = 0
    
    for i, utterance in enumerate(utterances):
        duration = float(utterance.get('speaking_duration', 0))
        if 'present new idea' in utterance.get('annotations', {}):
            # Find next decision
            search_time = current_time
            for j in range(i + 1, len(utterances)):
                next_duration = float(utterances[j].get('speaking_duration', 0))
                if 'propose decision' in utterances[j].get('annotations', {}):
                    decision_lags.append(search_time - current_time)
                    break
                search_time += next_duration
        current_time += duration
    
    if decision_lags:
        features['decision_lag'] = np.mean(decision_lags)
    
    # Calculate idea-agree-decision chain duration
    chain_durations = []
    current_time = 0
    
    for i in range(len(utterances) - 2):
        duration1 = float(utterances[i].get('speaking_duration', 0))
        duration2 = float(utterances[i+1].get('speaking_duration', 0))
        duration3 = float(utterances[i+2].get('speaking_duration', 0))
        
        if ('present new idea' in utterances[i].get('annotations', {}) and
            'express agreement' in utterances[i+1].get('annotations', {}) and
            'propose decision' in utterances[i+2].get('annotations', {})):
            chain_durations.append(duration1 + duration2 + duration3)
    
    if chain_durations:
        features['idea_agree_decision_chain_duration'] = np.mean(chain_durations)
    
    # Calculate burstiness of ideas using time windows
    window_size = 300  # 5-minute windows in seconds
    ideas_per_window = []
    current_window = 0
    window_ideas = 0
    
    for utterance in utterances:
        duration = float(utterance.get('speaking_duration', 0))
        if 'present new idea' in utterance.get('annotations', {}):
            window_ideas += 1
        current_window += duration
        if current_window >= window_size:
            ideas_per_window.append(window_ideas)
            current_window = 0
            window_ideas = 0
    
    if ideas_per_window:
        features['burstiness_of_ideas'] = np.std(ideas_per_window)
    
    # Calculate conflict resolution latency in seconds
    resolution_latencies = []
    current_time = 0
    
    for i, utterance in enumerate(utterances):
        duration = float(utterance.get('speaking_duration', 0))
        if 'reject idea' in utterance.get('annotations', {}):
            # Find next resolution
            search_time = current_time
            for j in range(i + 1, len(utterances)):
                next_duration = float(utterances[j].get('speaking_duration', 0))
                if 'resolve conflict' in utterances[j].get('annotations', {}):
                    resolution_latencies.append(search_time - current_time)
                    break
                search_time += next_duration
        current_time += duration
    
    if resolution_latencies:
        features['conflict_resolution_latency'] = np.mean(resolution_latencies)
    
    return features

def process_session_file(file_path):
    """Process a single session file and calculate all features."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    utterances = data.get('all_data', [])  # Changed from 'utterances' to 'all data'
    
    # Calculate all feature categories
    features = {}
    features.update(calculate_core_verbal_features(utterances))
    features.update(calculate_speaker_engagement_features(utterances))
    features.update(calculate_nonverbal_features(utterances))
    features.update(calculate_cognitive_load_features(utterances))
    features.update(calculate_gesture_features(utterances))
    features.update(calculate_turn_taking_features(utterances))
    features.update(calculate_screensharing_features(utterances))
    features.update(calculate_temporal_features(utterances))
    
    return features


def main():
    parser = argparse.ArgumentParser(description="Featurize session JSON files for a given dataset.")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., 2020NES)")
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = f'gemini_code/data/{dataset}/featurized data'
    os.makedirs(output_dir, exist_ok=True)
    
    data_dir = f'gemini_code/data/{dataset}'
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and filename != f'{dataset}_outcome.json':
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