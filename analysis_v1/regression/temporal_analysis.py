#!/usr/bin/env python3
"""
Temporal Analysis Script - Analyzing Predictive Relationships Across Meeting Segments

This script aggregates featurized data from beginning, middle, and end segments
and analyzes whether features at beginning predict features at middle and end.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_temporal_data(data_dir='data'):
    """
    Load and aggregate temporal data from featurized_with_when directories.
    
    Args:
        data_dir: Base directory containing conference data
    
    Returns:
        DataFrame with features from beginning, middle, and end segments
    """
    
    print("Loading temporal data from featurized_with_when directories...")
    
    all_sessions = []
    
    # Find all conference directories
    data_path = Path(data_dir)
    conf_dirs = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for conf_dir in conf_dirs:
        temporal_dir = conf_dir / 'featurized_with_when'
        
        if not temporal_dir.exists():
            print(f"  Skipping {conf_dir.name}: No featurized_with_when directory")
            continue
        
        print(f"  Processing {conf_dir.name}...")
        
        # Get all JSON files
        json_files = list(temporal_dir.glob('*.json'))
        
        # Group by session
        sessions = {}
        for json_file in json_files:
            filename = json_file.stem  # e.g., features_2020_11_05_NES_S1_beginning
            
            # Extract session and segment
            parts = filename.split('_')
            if len(parts) < 2:
                continue
            
            # Find segment (beginning, middle, end)
            segment = parts[-1] if parts[-1] in ['beginning', 'middle', 'end'] else None
            if not segment:
                continue
            
            # Session is everything except the segment
            session_id = '_'.join(parts[:-1])  # e.g., features_2020_11_05_NES_S1
            
            if session_id not in sessions:
                sessions[session_id] = {}
            
            # Load data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            sessions[session_id][segment] = data
        
        # Aggregate sessions that have all three segments
        for session_id, segments in sessions.items():
            if len(segments) == 3:  # Must have beginning, middle, and end
                session_data = {
                    'session': segments['beginning'].get('session', session_id.replace('features_', '')),
                    'conf_name': segments['beginning'].get('conf_name', conf_dir.name),
                    'num_members': segments['beginning'].get('num_members'),
                    'meeting_length': segments['beginning'].get('meeting_length'),
                }
                
                # Add features from each segment with prefix
                for segment in ['beginning', 'middle', 'end']:
                    for key, value in segments[segment].items():
                        # Skip metadata fields
                        if key in ['segment', 'conf_name', 'session', 'num_members', 'meeting_length']:
                            continue
                        
                        # Add with segment prefix
                        new_key = f"{segment}_{key}"
                        session_data[new_key] = value
                
                all_sessions.append(session_data)
            else:
                print(f"    Warning: {session_id} missing segments (has: {list(segments.keys())})")
    
    df = pd.DataFrame(all_sessions)
    print(f"\nLoaded {len(df)} sessions with complete temporal data (beginning, middle, end)")
    
    return df

def create_temporal_predictive_dataset(df, feature_categories=None):
    """
    Create datasets for temporal predictive analysis.
    
    Args:
        df: DataFrame with temporal features
        feature_categories: List of feature base names to analyze (e.g., ['num_evaluation_practices'])
    
    Returns:
        Dictionary with different analysis datasets
    """
    
    if feature_categories is None:
        # Extract unique feature names (without segment prefix)
        all_cols = df.columns
        feature_categories = set()
        for col in all_cols:
            for prefix in ['beginning_', 'middle_', 'end_']:
                if col.startswith(prefix):
                    feature_name = col.replace(prefix, '')
                    feature_categories.add(feature_name)
        feature_categories = sorted(list(feature_categories))
    
    print(f"\nIdentified {len(feature_categories)} feature categories for temporal analysis")
    
    datasets = {
        'beginning_to_middle': [],  # Beginning predicts middle
        'beginning_to_end': [],     # Beginning predicts end
        'middle_to_end': [],        # Middle predicts end
        'beginning_to_change': []   # Beginning predicts change (end - beginning)
    }
    
    # Check which features are available
    available_features = []
    for feature in feature_categories:
        has_all = (f'beginning_{feature}' in df.columns and 
                  f'middle_{feature}' in df.columns and 
                  f'end_{feature}' in df.columns)
        if has_all:
            available_features.append(feature)
    
    print(f"Features with complete temporal data: {len(available_features)}")
    
    return df, available_features

def run_temporal_regression(df, feature_name, predictor_segment='beginning', outcome_segment='middle', 
                           control_vars=None, normalize=True):
    """
    Run regression analysis for temporal prediction.
    
    Args:
        df: DataFrame with temporal data
        feature_name: Base feature name (without segment prefix)
        predictor_segment: Segment for predictor ('beginning' or 'middle')
        outcome_segment: Segment for outcome ('middle' or 'end')
        control_vars: List of control variables (without segment prefix)
        normalize: Whether to normalize features
    
    Returns:
        Dictionary with regression results
    """
    
    predictor_col = f"{predictor_segment}_{feature_name}"
    outcome_col = f"{outcome_segment}_{feature_name}"
    
    # Check if columns exist
    if predictor_col not in df.columns or outcome_col not in df.columns:
        return None
    
    # Prepare data
    required_cols = [predictor_col, outcome_col]
    
    # Add control variables from the predictor segment
    control_cols = []
    if control_vars:
        for ctrl in control_vars:
            ctrl_col = f"{predictor_segment}_{ctrl}"
            if ctrl_col in df.columns:
                control_cols.append(ctrl_col)
                required_cols.append(ctrl_col)
    
    # Also include session-level controls (no segment prefix)
    session_controls = ['num_members', 'meeting_length']
    for ctrl in session_controls:
        if ctrl in df.columns:
            control_cols.append(ctrl)
            required_cols.append(ctrl)
    
    valid_data = df[required_cols].dropna().copy()
    
    if len(valid_data) < 10:
        return None
    
    # Prepare X and y
    if control_cols:
        X = valid_data[[predictor_col] + control_cols].values
    else:
        X = valid_data[predictor_col].values.reshape(-1, 1)
    
    y = valid_data[outcome_col].values
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate metrics
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    n = len(valid_data)
    p = X.shape[1]
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Adjusted R-squared
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
    
    # Standard error and p-value for main coefficient
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    try:
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        var_cov_matrix = mse * XtX_inv
        se_coef = np.sqrt(var_cov_matrix[1, 1])
        
        t_stat = model.coef_[0] / se_coef if se_coef > 0 else np.nan
        df_resid = n - p - 1
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid)) if not np.isnan(t_stat) else np.nan
        
        ci_lower = model.coef_[0] - stats.t.ppf(0.975, df_resid) * se_coef
        ci_upper = model.coef_[0] + stats.t.ppf(0.975, df_resid) * se_coef
    except:
        se_coef = np.nan
        t_stat = np.nan
        p_value = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
    
    # Calculate correlation between predictor and outcome
    try:
        correlation = np.corrcoef(valid_data[predictor_col].values, valid_data[outcome_col].values)[0, 1]
    except:
        correlation = np.nan
    
    result = {
        'Feature': feature_name,
        'Predictor_Segment': predictor_segment,
        'Outcome_Segment': outcome_segment,
        'N': n,
        'Coefficient': model.coef_[0],
        'Standard_Error': se_coef,
        'T_Statistic': t_stat,
        'P_Value': p_value,
        'CI_Lower_95': ci_lower,
        'CI_Upper_95': ci_upper,
        'R_Squared': r_squared,
        'Adjusted_R_Squared': adj_r_squared,
        'RMSE': rmse,
        'Correlation': correlation,
        'Mean_Predictor': np.mean(valid_data[predictor_col]),
        'Std_Predictor': np.std(valid_data[predictor_col]),
        'Mean_Outcome': np.mean(valid_data[outcome_col]),
        'Std_Outcome': np.std(valid_data[outcome_col]),
        'Num_Controls': len(control_cols),
        'Controls': ', '.join(control_cols) if control_cols else 'None',
        'Normalized': normalize
    }
    
    return result

def run_change_score_analysis(df, feature_name, control_vars=None, normalize=True):
    """
    Analyze whether beginning features predict change from beginning to end.
    
    Args:
        df: DataFrame with temporal data
        feature_name: Base feature name
        control_vars: List of control variables
        normalize: Whether to normalize features
    
    Returns:
        Dictionary with regression results
    """
    
    beginning_col = f"beginning_{feature_name}"
    end_col = f"end_{feature_name}"
    
    if beginning_col not in df.columns or end_col not in df.columns:
        return None
    
    # Prepare required columns
    required_cols = [beginning_col, end_col]
    
    # Add control variables
    control_cols = []
    if control_vars:
        for ctrl in control_vars:
            ctrl_col = f"beginning_{ctrl}"
            if ctrl_col in df.columns:
                control_cols.append(ctrl_col)
                required_cols.append(ctrl_col)
    
    # Session-level controls
    session_controls = ['num_members', 'meeting_length']
    for ctrl in session_controls:
        if ctrl in df.columns:
            control_cols.append(ctrl)
            required_cols.append(ctrl)
    
    # Get valid data with all required columns
    valid_data = df[required_cols].dropna().copy()
    
    if len(valid_data) < 10:
        return None
    
    # Calculate change
    change = valid_data[end_col] - valid_data[beginning_col]
    
    # Predictor is beginning value
    if control_cols:
        X = valid_data[[beginning_col] + control_cols].values
    else:
        X = valid_data[beginning_col].values.reshape(-1, 1)
    
    y = change.values
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate metrics (similar to above)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    n = len(valid_data)
    p = X.shape[1]
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
    
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    try:
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        var_cov_matrix = mse * XtX_inv
        se_coef = np.sqrt(var_cov_matrix[1, 1])
        
        t_stat = model.coef_[0] / se_coef if se_coef > 0 else np.nan
        df_resid = n - p - 1
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid)) if not np.isnan(t_stat) else np.nan
        
        ci_lower = model.coef_[0] - stats.t.ppf(0.975, df_resid) * se_coef
        ci_upper = model.coef_[0] + stats.t.ppf(0.975, df_resid) * se_coef
    except:
        se_coef = np.nan
        t_stat = np.nan
        p_value = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
    
    try:
        correlation = np.corrcoef(valid_data[beginning_col].values, change.values)[0, 1]
    except:
        correlation = np.nan
    
    result = {
        'Feature': feature_name,
        'Analysis_Type': 'Beginning_Predicts_Change',
        'N': n,
        'Coefficient': model.coef_[0],
        'Standard_Error': se_coef,
        'T_Statistic': t_stat,
        'P_Value': p_value,
        'CI_Lower_95': ci_lower,
        'CI_Upper_95': ci_upper,
        'R_Squared': r_squared,
        'Adjusted_R_Squared': adj_r_squared,
        'RMSE': rmse,
        'Correlation': correlation,
        'Mean_Beginning': np.mean(valid_data[beginning_col]),
        'Mean_End': np.mean(valid_data[end_col]),
        'Mean_Change': np.mean(change),
        'Std_Change': np.std(change),
        'Num_Controls': len(control_cols),
        'Controls': ', '.join(control_cols) if control_cols else 'None'
    }
    
    return result

def save_temporal_results(results_dict, output_dir='regression'):
    """
    Save temporal analysis results to Excel files.
    
    Args:
        results_dict: Dictionary with different analysis results
        output_dir: Output directory
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for analysis_type, results in results_dict.items():
        if not results:
            continue
        
        df = pd.DataFrame(results)
        filename = f"{output_dir}/temporal_{analysis_type}_results.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # All results
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Significant results
            if 'P_Value' in df.columns:
                significant = df[df['P_Value'] < 0.05].copy()
                significant = significant.sort_values('P_Value')
                significant.to_excel(writer, sheet_name='Significant_Results', index=False)
            
            # High R-squared results
            if 'R_Squared' in df.columns:
                high_r2 = df[df['R_Squared'] > 0.1].copy()
                high_r2 = high_r2.sort_values('R_Squared', ascending=False)
                high_r2.to_excel(writer, sheet_name='High_R_Squared', index=False)
        
        print(f"Saved: {filename}")

def main():
    """
    Main function to run temporal analysis.
    """
    
    print("="*60)
    print("TEMPORAL PREDICTIVE ANALYSIS")
    print("="*60)
    
    # Load temporal data (adjust path relative to script location)
    # Get the script directory and navigate to data folder
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'  # Goes from regression/ to gemini_code/data/
    df = load_temporal_data(str(data_dir))
    
    if df.empty:
        print("No temporal data found!")
        return
    
    # Create temporal datasets
    df, available_features = create_temporal_predictive_dataset(df)
    
    print(f"\nAvailable features for analysis:")
    for i, feature in enumerate(available_features[:20], 1):
        print(f"  {i}. {feature}")
    if len(available_features) > 20:
        print(f"  ... and {len(available_features) - 20} more")
    
    # Define control variables
    control_vars = ['num_facilitator', 'total_utterances']
    
    # Run analyses
    results_dict = {
        'beginning_to_middle': [],
        'beginning_to_end': [],
        'middle_to_end': [],
        'change_score': []
    }
    
    print("\n" + "="*60)
    print("Running temporal regressions...")
    print("="*60)
    
    for feature in available_features:
        # Skip control variables and metadata
        skip_features = ['num_members', 'meeting_length', 'segment_length', 'num_segment_members']
        # Also skip any features that are in control_vars
        if control_vars:
            skip_features.extend(control_vars)
        
        if feature in skip_features:
            continue
        
        print(f"\nAnalyzing: {feature}")
        
        # Beginning → Middle
        result = run_temporal_regression(df, feature, 'beginning', 'middle', control_vars, normalize=True)
        if result:
            results_dict['beginning_to_middle'].append(result)
            print(f"  Beginning → Middle: R² = {result['R_Squared']:.4f}, p = {result['P_Value']:.4f}")
        
        # Beginning → End
        result = run_temporal_regression(df, feature, 'beginning', 'end', control_vars, normalize=True)
        if result:
            results_dict['beginning_to_end'].append(result)
            print(f"  Beginning → End: R² = {result['R_Squared']:.4f}, p = {result['P_Value']:.4f}")
        
        # Middle → End
        result = run_temporal_regression(df, feature, 'middle', 'end', control_vars, normalize=True)
        if result:
            results_dict['middle_to_end'].append(result)
            print(f"  Middle → End: R² = {result['R_Squared']:.4f}, p = {result['P_Value']:.4f}")
        
        # Change score analysis
        result = run_change_score_analysis(df, feature, control_vars, normalize=True)
        if result:
            results_dict['change_score'].append(result)
            print(f"  Beginning → Change: R² = {result['R_Squared']:.4f}, p = {result['P_Value']:.4f}")
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    save_temporal_results(results_dict)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for analysis_type, results in results_dict.items():
        if not results:
            continue
        
        df_results = pd.DataFrame(results)
        significant = df_results[df_results['P_Value'] < 0.05] if 'P_Value' in df_results.columns else pd.DataFrame()
        
        print(f"\n{analysis_type.replace('_', ' ').title()}:")
        print(f"  Total analyses: {len(df_results)}")
        print(f"  Significant (p < 0.05): {len(significant)}")
        if 'R_Squared' in df_results.columns:
            print(f"  Mean R²: {df_results['R_Squared'].mean():.4f}")
            print(f"  Max R²: {df_results['R_Squared'].max():.4f}")
        
        if len(significant) > 0 and 'R_Squared' in significant.columns:
            best = significant.loc[significant['R_Squared'].idxmax()]
            print(f"  Best feature: {best['Feature']} (R² = {best['R_Squared']:.4f}, p = {best['P_Value']:.4f})")

if __name__ == "__main__":
    main()

