#!/usr/bin/env python3
"""
Regression Analysis Script

This script runs individual regressions between each feature column and outcome variables,
saving detailed results to an Excel file.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def run_individual_regressions(df, feature_cols, outcome_vars=['num_teams', 'num_funded_teams'], control_vars=None):
    """
    Run individual regressions for each feature against each outcome variable.
    
    Args:
        df: DataFrame containing the data
        feature_cols: List of feature column names
        outcome_vars: List of outcome variable names
        control_vars: List of control variable names (optional)
    
    Returns:
        DataFrame with detailed regression results
    """
    
    results = []
    
    # Handle control variables
    if control_vars is None:
        control_vars = []
    else:
        # Filter out control variables from feature columns to avoid double inclusion
        feature_cols = [col for col in feature_cols if col not in control_vars]
        print(f"Control variables: {control_vars}")
        print(f"Features (excluding controls): {len(feature_cols)}")
    
    print(f"Running regressions for {len(feature_cols)} features against {len(outcome_vars)} outcomes...")
    if control_vars:
        print(f"With {len(control_vars)} control variables included in all models")
    print("="*60)
    
    for feature in feature_cols:
        print(f"\nProcessing feature: {feature}")
        
        for outcome in outcome_vars:
            print(f"  -> {outcome}")
            
            # Prepare data - include control variables if specified
            required_cols = [feature, outcome]
            if control_vars and len(control_vars) > 0:
                required_cols.extend(control_vars)
            
            # Check if all required columns exist in the dataframe
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"    Skipping {feature} vs {outcome}: Missing columns {missing_cols}")
                continue
            
            valid_data = df[required_cols].dropna()
            
            # Need more data points when including control variables
            min_points = 3 + len(control_vars) if control_vars else 3
            if len(valid_data) < min_points:
                print(f"    Skipping {feature} vs {outcome}: Insufficient data ({len(valid_data)} points, need {min_points})")
                continue
            
            # Prepare X matrix with feature and control variables
            if control_vars:
                X = valid_data[[feature] + control_vars].values
            else:
                X = valid_data[feature].values.reshape(-1, 1)
            
            y = valid_data[outcome].values
            
            # Fit the regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            r_squared = r2_score(y, y_pred)
            n = len(valid_data)
            p = 1 + len(control_vars)  # number of predictors (feature + controls)
            
            # Calculate adjusted R-squared
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            
            # Calculate residuals and standard error
            residuals = y - y_pred
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            
            # Calculate standard error of coefficients for the main feature
            # For multiple regression, we need the variance-covariance matrix
            try:
                # Calculate the variance-covariance matrix
                X_with_intercept = np.column_stack([np.ones(n), X])
                XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                var_cov_matrix = mse * XtX_inv
                
                # Standard error for the main feature (first predictor after intercept)
                se_coef = np.sqrt(var_cov_matrix[1, 1])
                
                # Calculate t-statistic and p-value for the main feature coefficient
                t_stat = model.coef_[0] / se_coef if se_coef > 0 else np.nan
                degrees_of_freedom = n - p - 1  # degrees of freedom
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), degrees_of_freedom)) if not np.isnan(t_stat) else np.nan
                
                # Calculate confidence intervals (95%)
                alpha = 0.05
                t_critical = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
                ci_lower = model.coef_[0] - t_critical * se_coef
                ci_upper = model.coef_[0] + t_critical * se_coef
                
            except np.linalg.LinAlgError:
                # Fallback for singular matrix
                se_coef = np.nan
                t_stat = np.nan
                p_value = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
            
            # Calculate F-statistic and its p-value
            f_stat = (r_squared / (1 - r_squared)) * (n - p - 1) / p if r_squared < 1 else np.inf
            f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1) if not np.isinf(f_stat) else 0
            
            # Store results
            result = {
                'Feature': feature,
                'Outcome': outcome,
                'N': n,
                'Intercept': model.intercept_,
                'Coefficient': model.coef_[0],
                'Standard_Error': se_coef,
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'R_Squared': r_squared,
                'Adjusted_R_Squared': adj_r_squared,
                'RMSE': rmse,
                'CI_Lower_95': ci_lower,
                'CI_Upper_95': ci_upper,
                'F_Statistic': f_stat,
                'F_P_Value': f_p_value,
                'Mean_X': np.mean(X[:, 0]),  # Mean of main feature
                'Std_X': np.std(X[:, 0]),    # Std of main feature
                'Mean_Y': np.mean(y),
                'Std_Y': np.std(y),
                'Num_Control_Vars': len(control_vars),
                'Control_Vars': ', '.join(control_vars) if control_vars else 'None'
            }
            
            results.append(result)
            
            print(f"    R² = {r_squared:.4f}, p = {p_value:.4f}, β = {model.coef_[0]:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\nCompleted {len(results_df)} regressions")
    return results_df

def save_results_to_excel(results_df, filename):
    """
    Save regression results to an Excel file with multiple sheets.
    
    Args:
        results_df: DataFrame with regression results
        filename: Output filename
    """
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main results sheet
        results_df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Summary statistics
        summary_stats = results_df.groupby('Outcome').agg({
            'R_Squared': ['mean', 'std', 'min', 'max'],
            'P_Value': ['mean', 'std', 'min', 'max'],
            'Coefficient': ['mean', 'std', 'min', 'max'],
            'N': ['mean', 'std', 'min', 'max']
        }).round(4)
        summary_stats.to_excel(writer, sheet_name='Summary_Stats')
        
        # Significant results only (p < 0.05)
        significant = results_df[results_df['P_Value'] < 0.05].copy()
        significant = significant.sort_values('P_Value')
        significant.to_excel(writer, sheet_name='Significant_Results', index=False)
        
        # Results by outcome
        for outcome in results_df['Outcome'].unique():
            outcome_data = results_df[results_df['Outcome'] == outcome].copy()
            outcome_data = outcome_data.sort_values('R_Squared', ascending=False)
            sheet_name = f'Results_{outcome}'
            outcome_data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Results saved to {filename}")
    print(f"  - All_Results: Complete results")
    print(f"  - Summary_Stats: Summary statistics by outcome")
    print(f"  - Significant_Results: Only significant results (p < 0.05)")
    print(f"  - Results_[outcome]: Results grouped by outcome variable")



def main():
    """
    Main function to run the regression analysis.
    """
    
    # Load your data - adjust this path as needed
    try:
        # Try to load the data - you may need to adjust this path
        df = pd.read_excel('data/all_data_df.xlsx')
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print("Data file not found. Please ensure 'data/all_data_df.xlsx' exists.")
        print("Or modify the file path in the script.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Define feature columns - adjust these based on your actual column names
    # You may need to modify this list based on your actual data
    feature_cols = [col for col in df.columns if col not in ['num_teams', 'num_funded_teams', 'conference_name', 'session_id', 'has_teams', 'has_funded_teams']]
    
    # Remove any non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col in numeric_cols]
    
    print(f"Found {len(feature_cols)} feature columns:")
    for i, col in enumerate(feature_cols[:10]):  # Show first 10
        print(f"  {i+1}. {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")
    
    
    # Define control variables (modify this list as needed)
    # Example: control_vars = ['conference_year', 'session_duration']
    # You can use suggested_controls or define your own list
    control_vars = ['meeting_length', 'num_members', 'num_facilitator']  # Add your control variables here
   
    # Run regressions
    results_df = run_individual_regressions(df, feature_cols, outcome_vars=['num_teams', 'num_funded_teams'], control_vars=control_vars)
    
    # Save results
    save_results_to_excel(results_df, 'regression/linear_regression_results_with_controls.xlsx')
    
    # Print summary
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS SUMMARY")
    print("="*60)
    
    for outcome in results_df['Outcome'].unique():
        outcome_results = results_df[results_df['Outcome'] == outcome]
        significant = outcome_results[outcome_results['P_Value'] < 0.05]
        
        print(f"\n{outcome}:")
        print(f"  Total regressions: {len(outcome_results)}")
        print(f"  Significant (p < 0.05): {len(significant)}")
        print(f"  Mean R²: {outcome_results['R_Squared'].mean():.4f}")
        print(f"  Max R²: {outcome_results['R_Squared'].max():.4f}")
        
        if len(significant) > 0:
            best_feature = significant.loc[significant['R_Squared'].idxmax()]
            print(f"  Best feature: {best_feature['Feature']} (R² = {best_feature['R_Squared']:.4f}, p = {best_feature['P_Value']:.4f})")

if __name__ == "__main__":
    main()
