#!/usr/bin/env python3
"""
Negative Binomial Regression Analysis Script

This script runs individual negative binomial regressions between each feature column and count outcome variables,
saving detailed results to an Excel file. Perfect for count data with overdispersion.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
import warnings
warnings.filterwarnings('ignore')

def analyze_outcome_distributions(df, outcome_vars=['num_teams', 'num_funded_teams']):
    """
    Analyze the distribution of outcome variables to recommend appropriate regression models.
    
    Args:
        df: DataFrame containing the data
        outcome_vars: List of potential outcome variable names
    
    Returns:
        Dictionary with analysis results and recommendations
    """
    
    print("=== OUTCOME VARIABLE DISTRIBUTION ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    
    available_outcomes = [var for var in outcome_vars if var in df.columns]
    print(f"Available outcome variables: {available_outcomes}")
    
    analysis_results = {}
    
    for var in available_outcomes:
        print(f"\n--- {var} ---")
        print(f"Data type: {df[var].dtype}")
        print(f"Unique values: {sorted(df[var].unique())}")
        print(f"Value counts:")
        value_counts = df[var].value_counts().sort_index()
        print(value_counts)
        
        mean_val = df[var].mean()
        std_val = df[var].std()
        min_val = df[var].min()
        max_val = df[var].max()
        zero_count = (df[var] == 0).sum()
        zero_pct = (df[var] == 0).mean() * 100
        
        print(f"Mean: {mean_val:.3f}")
        print(f"Std: {std_val:.3f}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Zero values: {zero_count} ({zero_pct:.1f}%)")
        
        # Check for overdispersion (variance > mean for count data)
        variance = df[var].var()
        overdispersion = variance > mean_val if mean_val > 0 else False
        overdispersion_ratio = variance / mean_val if mean_val > 0 else np.inf
        
        print(f"Variance: {variance:.3f}")
        print(f"Overdispersion ratio (variance/mean): {overdispersion_ratio:.3f}")
        print(f"Overdispersed: {overdispersion}")
        
        # Determine variable type and recommend model
        unique_vals = df[var].nunique()
        is_integer = df[var].dtype in ['int64', 'int32', 'int16', 'int8']
        is_binary = unique_vals == 2 and set(df[var].unique()).issubset({0, 1})
        is_count = is_integer and min_val >= 0 and max_val > 0
        
        # Model recommendations
        if is_binary:
            model_type = "Binary Logistic Regression"
            reason = "Binary outcome (0/1 values)"
        elif is_count and overdispersion:
            model_type = "Negative Binomial Regression"
            reason = f"Count data with overdispersion (ratio: {overdispersion_ratio:.3f})"
        elif is_count and zero_pct > 20:
            model_type = "Zero-Inflated Negative Binomial"
            reason = f"Count data with high zero inflation ({zero_pct:.1f}% zeros) and overdispersion"
        elif is_count:
            model_type = "Poisson Regression"
            reason = "Count data without overdispersion"
        elif is_integer and min_val >= 0:
            model_type = "Ordinal Logistic Regression"
            reason = "Ordinal categorical data"
        else:
            model_type = "Linear Regression"
            reason = "Continuous outcome variable"
        
        print(f"Recommended model: {model_type}")
        print(f"Reason: {reason}")
        
        analysis_results[var] = {
            'data_type': df[var].dtype,
            'unique_values': unique_vals,
            'is_binary': is_binary,
            'is_count': is_count,
            'is_integer': is_integer,
            'zero_percentage': zero_pct,
            'mean': mean_val,
            'std': std_val,
            'variance': variance,
            'overdispersion_ratio': overdispersion_ratio,
            'is_overdispersed': overdispersion,
            'min': min_val,
            'max': max_val,
            'recommended_model': model_type,
            'reason': reason
        }
    
    return analysis_results

def run_individual_negative_binomial_regressions(df, feature_cols, outcome_vars=['num_teams', 'num_funded_teams'], control_vars=None, normalize=True):
    """
    Run individual negative binomial regressions for each feature against each count outcome variable.
    
    Args:
        df: DataFrame containing the data
        feature_cols: List of feature column names
        outcome_vars: List of count outcome variable names
        control_vars: List of control variable names (optional)
        normalize: Boolean, whether to normalize features (default: True)
    
    Returns:
        DataFrame with detailed negative binomial regression results
    """
    
    results = []
    
    # Initialize scaler for normalization
    scaler = StandardScaler() if normalize else None
    
    print(f"Running negative binomial regressions for {len(feature_cols)} features against {len(outcome_vars)} outcomes...")
    if control_vars:
        print(f"Control variables: {control_vars}")
    if normalize:
        print("Features will be normalized (standardized)")
    print("="*60)
    
    for feature in feature_cols:
        print(f"\nProcessing feature: {feature}")
        
        for outcome in outcome_vars:
            print(f"  -> {outcome}")
            
            # Prepare data - include control variables if specified
            if control_vars:
                # Check if all control variables exist
                missing_controls = [var for var in control_vars if var not in df.columns]
                if missing_controls:
                    print(f"    Skipping {feature} vs {outcome}: Missing control variables: {missing_controls}")
                    continue
                
                # Include feature, outcome, and control variables
                all_vars = [feature, outcome] + control_vars
                valid_data = df[all_vars].dropna()
            else:
                valid_data = df[[feature, outcome]].dropna()
            
            if len(valid_data) < 10:  # Need sufficient data for negative binomial regression
                print(f"    Skipping {feature} vs {outcome}: Insufficient data ({len(valid_data)} points)")
                continue
            
            # Check if outcome has sufficient variation
            if valid_data[outcome].var() == 0:
                print(f"    Skipping {feature} vs {outcome}: No variation in outcome")
                continue
            
            # Prepare features (main feature + control variables)
            if control_vars:
                X = valid_data[[feature] + control_vars].values
            else:
                X = valid_data[feature].values.reshape(-1, 1)
            
            y = valid_data[outcome].values
            
            # Normalize features if requested
            if normalize:
                X = scaler.fit_transform(X)
            
            # Add constant term for statsmodels
            X_with_const = sm.add_constant(X)
            
            # Fit the negative binomial regression model
            try:
                # Use NB2 model (most common)
                model = NegativeBinomial(y, X_with_const, loglike_method='nb2')
                result = model.fit(disp=0, maxiter=1000)
                
                # Extract results
                n = len(y)
                p = X.shape[1]  # number of predictors (main feature + controls)
                
                # Get coefficient for main feature (first predictor after constant)
                coef = result.params[1]  # First coefficient after constant
                se_coef = result.bse[1]   # Standard error
                z_stat = result.tvalues[1]  # Z-statistic
                p_value = result.pvalues[1]  # P-value
                
                # Calculate confidence intervals (95%)
                ci_lower = coef - 1.96 * se_coef
                ci_upper = coef + 1.96 * se_coef
                
                # Calculate pseudo R-squared (McFadden's)
                null_model = NegativeBinomial(y, np.ones((n, 1)), loglike_method='nb2')
                null_result = null_model.fit(disp=0, maxiter=1000)
                
                mcfadden_r2 = 1 - (result.llf / null_result.llf)
                
                # Calculate AIC and BIC
                aic = result.aic
                bic = result.bic
                
                # Calculate log-likelihood
                log_likelihood = result.llf
                
                # Calculate alpha parameter (overdispersion parameter)
                alpha = result.params[-1] if len(result.params) > p + 1 else np.nan
                
                # Calculate incidence rate ratio (IRR) and its CI
                irr = np.exp(coef)
                irr_ci_lower = np.exp(ci_lower)
                irr_ci_upper = np.exp(ci_upper)
                
                # Calculate predicted values and residuals
                y_pred = result.predict()
                residuals = y - y_pred
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean(residuals**2))
                
                # Calculate mean absolute error
                mae = np.mean(np.abs(residuals))
                
                # Store results
                result_dict = {
                    'Feature': feature,
                    'Outcome': outcome,
                    'N': n,
                    'Intercept': result.params[0],
                    'Coefficient': coef,
                    'Standard_Error': se_coef,
                    'Z_Statistic': z_stat,
                    'P_Value': p_value,
                    'IRR': irr,
                    'IRR_CI_Lower_95': irr_ci_lower,
                    'IRR_CI_Upper_95': irr_ci_upper,
                    'CI_Lower_95': ci_lower,
                    'CI_Upper_95': ci_upper,
                    'McFadden_R_Squared': mcfadden_r2,
                    'Log_Likelihood': log_likelihood,
                    'AIC': aic,
                    'BIC': bic,
                    'Alpha_Overdispersion': alpha,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Mean_X': np.mean(X[:, 0]),  # Mean of main feature
                    'Std_X': np.std(X[:, 0]),    # Std of main feature
                    'Mean_Y': np.mean(y),
                    'Std_Y': np.std(y),
                    'Positive_Class_Count': np.sum(y > 0),
                    'Zero_Count': np.sum(y == 0),
                    'Num_Control_Vars': len(control_vars) if control_vars else 0,
                    'Control_Vars': ', '.join(control_vars) if control_vars else 'None',
                    'Normalized': normalize
                }
                
                results.append(result_dict)
                
                print(f"    IRR = {irr:.4f}, p = {p_value:.4f}, α = {alpha:.4f}, R² = {mcfadden_r2:.4f}")
                
            except Exception as e:
                print(f"    Error fitting {feature} vs {outcome}: {str(e)}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\nCompleted {len(results_df)} negative binomial regressions")
    return results_df

def save_results_to_excel(results_df, filename):
    """
    Save negative binomial regression results to an Excel file with multiple sheets.
    
    Args:
        results_df: DataFrame with regression results
        filename: Output filename
    """
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main results sheet
        results_df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Summary statistics
        summary_stats = results_df.groupby('Outcome').agg({
            'McFadden_R_Squared': ['mean', 'std', 'min', 'max'],
            'P_Value': ['mean', 'std', 'min', 'max'],
            'IRR': ['mean', 'std', 'min', 'max'],
            'Alpha_Overdispersion': ['mean', 'std', 'min', 'max'],
            'N': ['mean', 'std', 'min', 'max']
        }).round(4)
        summary_stats.to_excel(writer, sheet_name='Summary_Stats')
        
        # Significant results only (p < 0.05)
        significant = results_df[results_df['P_Value'] < 0.05].copy()
        significant = significant.sort_values('P_Value')
        significant.to_excel(writer, sheet_name='Significant_Results', index=False)
        
        # High performance results (McFadden R² > 0.1)
        high_performance = results_df[results_df['McFadden_R_Squared'] > 0.1].copy()
        high_performance = high_performance.sort_values('McFadden_R_Squared', ascending=False)
        high_performance.to_excel(writer, sheet_name='High_Performance', index=False)
        
        # Results by outcome
        for outcome in results_df['Outcome'].unique():
            outcome_data = results_df[results_df['Outcome'] == outcome].copy()
            outcome_data = outcome_data.sort_values('McFadden_R_Squared', ascending=False)
            sheet_name = f'Results_{outcome}'
            outcome_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # IRR analysis
        irr_analysis = results_df[['Feature', 'Outcome', 'IRR', 'IRR_CI_Lower_95', 'IRR_CI_Upper_95', 'P_Value']].copy()
        irr_analysis = irr_analysis.sort_values('IRR', ascending=False)
        irr_analysis.to_excel(writer, sheet_name='IRR_Analysis', index=False)
    
    print(f"Results saved to {filename}")
    print(f"  - All_Results: Complete results")
    print(f"  - Summary_Stats: Summary statistics by outcome")
    print(f"  - Significant_Results: Only significant results (p < 0.05)")
    print(f"  - High_Performance: High performing models (McFadden R² > 0.1)")
    print(f"  - Results_[outcome]: Results grouped by outcome variable")
    print(f"  - IRR_Analysis: Incidence Rate Ratio analysis")

def main():
    """
    Main function to run the negative binomial regression analysis.
    """
    
    # Load your data - adjust this path as needed
    try:
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
    feature_cols = [col for col in df.columns if col not in 
                    ['num_teams', 'num_funded_teams', 'conference_name', 'session_id', 
                     'has_teams', 'has_funded_teams']]
    
    # Remove any non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col in numeric_cols]
    
    print(f"Found {len(feature_cols)} feature columns:")
    for i, col in enumerate(feature_cols[:10]):  # Show first 10
        print(f"  {i+1}. {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")
    
    # Analyze outcome variable distributions to recommend appropriate models
    print("\n" + "="*60)
    outcome_analysis = analyze_outcome_distributions(df)
    print("="*60)
    
    # Check if count outcome variables exist
    outcome_vars = ['num_teams', 'num_funded_teams']
    available_outcomes = [var for var in outcome_vars if var in df.columns]
    
    if not available_outcomes:
        print("No count outcome variables found. Please check your data.")
        print("Available outcome variables:", [var for var in df.columns if 'team' in var.lower() or 'fund' in var.lower()])
        return
    
    print(f"Using count outcome variables: {available_outcomes}")
    
    # Define control variables (modify this list as needed)
    control_vars = ['meeting_length', 'num_members', 'num_facilitator', 'total_utterances']
    
    # Check if control variables exist in the dataset
    available_controls = [var for var in control_vars if var in df.columns]
    if available_controls != control_vars:
        missing_controls = [var for var in control_vars if var not in df.columns]
        print(f"Warning: Some control variables not found: {missing_controls}")
        print(f"Using available control variables: {available_controls}")
        control_vars = available_controls
    
    # Run negative binomial regressions with control variables and normalization
    results_df = run_individual_negative_binomial_regressions(df, feature_cols, available_outcomes, 
                                                           control_vars=control_vars, normalize=True)
    
    if results_df.empty:
        print("No valid negative binomial regressions could be performed.")
        return
    
    # Save results
    save_results_to_excel(results_df, 'regression/negative_binomial_regression_results.xlsx')
    
    # Print summary
    print("\n" + "="*60)
    print("NEGATIVE BINOMIAL REGRESSION ANALYSIS SUMMARY")
    print("="*60)
    
    for outcome in results_df['Outcome'].unique():
        outcome_results = results_df[results_df['Outcome'] == outcome]
        significant = outcome_results[outcome_results['P_Value'] < 0.05]
        high_performance = outcome_results[outcome_results['McFadden_R_Squared'] > 0.1]
        
        print(f"\n{outcome}:")
        print(f"  Total regressions: {len(outcome_results)}")
        print(f"  Significant (p < 0.05): {len(significant)}")
        print(f"  High performance (McFadden R² > 0.1): {len(high_performance)}")
        print(f"  Mean McFadden R²: {outcome_results['McFadden_R_Squared'].mean():.4f}")
        print(f"  Max McFadden R²: {outcome_results['McFadden_R_Squared'].max():.4f}")
        print(f"  Mean Alpha (overdispersion): {outcome_results['Alpha_Overdispersion'].mean():.4f}")
        
        if len(significant) > 0:
            best_feature = significant.loc[significant['McFadden_R_Squared'].idxmax()]
            print(f"  Best significant feature: {best_feature['Feature']} (McFadden R² = {best_feature['McFadden_R_Squared']:.4f}, p = {best_feature['P_Value']:.4f})")
        
        if len(high_performance) > 0:
            best_performance = high_performance.loc[high_performance['McFadden_R_Squared'].idxmax()]
            print(f"  Best performing feature: {best_performance['Feature']} (McFadden R² = {best_performance['McFadden_R_Squared']:.4f})")

if __name__ == "__main__":
    main()
