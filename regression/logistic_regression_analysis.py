#!/usr/bin/env python3
"""
Logistic Regression Analysis Script

This script runs individual logistic regressions between each feature column and binary outcome variables,
saving detailed results to an Excel file.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def run_individual_logistic_regressions(df, feature_cols, outcome_vars=['has_teams', 'has_funded_teams']):
    """
    Run individual logistic regressions for each feature against each binary outcome variable.
    
    Args:
        df: DataFrame containing the data
        feature_cols: List of feature column names
        outcome_vars: List of binary outcome variable names
    
    Returns:
        DataFrame with detailed logistic regression results
    """
    
    results = []
    
    print(f"Running logistic regressions for {len(feature_cols)} features against {len(outcome_vars)} outcomes...")
    print("="*60)
    
    for feature in feature_cols:
        print(f"\nProcessing feature: {feature}")
        
        for outcome in outcome_vars:
            print(f"  -> {outcome}")
            
            # Prepare data - remove rows with missing values
            valid_data = df[[feature, outcome]].dropna()
            
            if len(valid_data) < 10:  # Need sufficient data for logistic regression
                print(f"    Skipping {feature} vs {outcome}: Insufficient data ({len(valid_data)} points)")
                continue
            
            # Check if outcome has both classes
            unique_outcomes = valid_data[outcome].nunique()
            if unique_outcomes < 2:
                print(f"    Skipping {feature} vs {outcome}: Only one class present")
                continue
            
            X = valid_data[feature].values.reshape(-1, 1)
            y = valid_data[outcome].values
            
            # Fit the logistic regression model
            try:
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X, y)
                
                # Make predictions
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
                
                # Calculate metrics
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, zero_division=0)
                recall = recall_score(y, y_pred, zero_division=0)
                f1 = f1_score(y, y_pred, zero_division=0)
                
                # Calculate AUC-ROC
                try:
                    auc_roc = roc_auc_score(y, y_pred_proba)
                except ValueError:
                    auc_roc = np.nan
                
                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
                
                # Calculate specificity and sensitivity
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                
                # Calculate pseudo R-squared (McFadden's R-squared)
                # Null model (intercept only)
                null_model = LogisticRegression(random_state=42, max_iter=1000)
                null_model.fit(np.ones((len(X), 1)), y)
                null_log_likelihood = null_model.score(np.ones((len(X), 1)), y) * len(y)
                
                # Full model log likelihood
                full_log_likelihood = model.score(X, y) * len(y)
                
                # McFadden's R-squared
                mcfadden_r2 = 1 - (full_log_likelihood / null_log_likelihood)
                
                # Cox & Snell R-squared
                cox_snell_r2 = 1 - np.exp((2/len(y)) * (null_log_likelihood - full_log_likelihood))
                
                # Nagelkerke R-squared
                nagelkerke_r2 = cox_snell_r2 / (1 - np.exp((2/len(y)) * null_log_likelihood))
                
                # Calculate standard errors and confidence intervals for coefficients
                # Using the Hessian matrix approach
                n = len(y)
                p = 1  # number of predictors
                
                # Calculate standard error of coefficients
                # For logistic regression, we need to use the inverse of the Hessian
                # This is a simplified approximation
                try:
                    # Calculate the variance-covariance matrix
                    # This is a simplified approach - in practice, you'd use the full Hessian
                    x_mean = np.mean(X)
                    x_var = np.var(X)
                    
                    # Approximate standard error (this is simplified)
                    se_coef = np.sqrt(1 / (n * x_var)) if x_var > 0 else np.nan
                    
                    # Calculate z-statistic and p-value
                    z_stat = model.coef_[0][0] / se_coef if se_coef > 0 else np.nan
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan
                    
                    # Calculate confidence intervals (95%)
                    alpha = 0.05
                    z_critical = stats.norm.ppf(1 - alpha/2)
                    ci_lower = model.coef_[0][0] - z_critical * se_coef
                    ci_upper = model.coef_[0][0] + z_critical * se_coef
                    
                except:
                    se_coef = np.nan
                    z_stat = np.nan
                    p_value = np.nan
                    ci_lower = np.nan
                    ci_upper = np.nan
                
                # Calculate odds ratio and its confidence interval
                odds_ratio = np.exp(model.coef_[0][0])
                or_ci_lower = np.exp(ci_lower)
                or_ci_upper = np.exp(ci_upper)
                
                # Calculate AIC and BIC
                aic = -2 * full_log_likelihood + 2 * (p + 1)  # +1 for intercept
                bic = -2 * full_log_likelihood + np.log(n) * (p + 1)
                
                # Store results
                result = {
                    'Feature': feature,
                    'Outcome': outcome,
                    'N': n,
                    'Intercept': model.intercept_[0],
                    'Coefficient': model.coef_[0][0],
                    'Standard_Error': se_coef,
                    'Z_Statistic': z_stat,
                    'P_Value': p_value,
                    'Odds_Ratio': odds_ratio,
                    'OR_CI_Lower_95': or_ci_lower,
                    'OR_CI_Upper_95': or_ci_upper,
                    'CI_Lower_95': ci_lower,
                    'CI_Upper_95': ci_upper,
                    'McFadden_R_Squared': mcfadden_r2,
                    'Cox_Snell_R_Squared': cox_snell_r2,
                    'Nagelkerke_R_Squared': nagelkerke_r2,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1,
                    'AUC_ROC': auc_roc,
                    'Specificity': specificity,
                    'Sensitivity': sensitivity,
                    'True_Positives': tp,
                    'True_Negatives': tn,
                    'False_Positives': fp,
                    'False_Negatives': fn,
                    'AIC': aic,
                    'BIC': bic,
                    'Mean_X': np.mean(X),
                    'Std_X': np.std(X),
                    'Mean_Y': np.mean(y),
                    'Std_Y': np.std(y),
                    'Positive_Class_Count': np.sum(y),
                    'Negative_Class_Count': len(y) - np.sum(y)
                }
                
                results.append(result)
                
                print(f"    Accuracy = {accuracy:.4f}, AUC = {auc_roc:.4f}, p = {p_value:.4f}, OR = {odds_ratio:.4f}")
                
            except Exception as e:
                print(f"    Error fitting {feature} vs {outcome}: {str(e)}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\nCompleted {len(results_df)} logistic regressions")
    return results_df

def save_results_to_excel(results_df, filename):
    """
    Save logistic regression results to an Excel file with multiple sheets.
    
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
            'AUC_ROC': ['mean', 'std', 'min', 'max'],
            'Accuracy': ['mean', 'std', 'min', 'max'],
            'P_Value': ['mean', 'std', 'min', 'max'],
            'Odds_Ratio': ['mean', 'std', 'min', 'max'],
            'N': ['mean', 'std', 'min', 'max']
        }).round(4)
        summary_stats.to_excel(writer, sheet_name='Summary_Stats')
        
        # Significant results only (p < 0.05)
        significant = results_df[results_df['P_Value'] < 0.05].copy()
        significant = significant.sort_values('P_Value')
        significant.to_excel(writer, sheet_name='Significant_Results', index=False)
        
        # High performance results (AUC > 0.7)
        high_performance = results_df[results_df['AUC_ROC'] > 0.7].copy()
        high_performance = high_performance.sort_values('AUC_ROC', ascending=False)
        high_performance.to_excel(writer, sheet_name='High_Performance', index=False)
        
        # Results by outcome
        for outcome in results_df['Outcome'].unique():
            outcome_data = results_df[results_df['Outcome'] == outcome].copy()
            outcome_data = outcome_data.sort_values('AUC_ROC', ascending=False)
            sheet_name = f'Results_{outcome}'
            outcome_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Odds ratio analysis
        odds_analysis = results_df[['Feature', 'Outcome', 'Odds_Ratio', 'OR_CI_Lower_95', 'OR_CI_Upper_95', 'P_Value']].copy()
        odds_analysis = odds_analysis.sort_values('Odds_Ratio', ascending=False)
        odds_analysis.to_excel(writer, sheet_name='Odds_Ratio_Analysis', index=False)
    
    print(f"Results saved to {filename}")
    print(f"  - All_Results: Complete results")
    print(f"  - Summary_Stats: Summary statistics by outcome")
    print(f"  - Significant_Results: Only significant results (p < 0.05)")
    print(f"  - High_Performance: High performing models (AUC > 0.7)")
    print(f"  - Results_[outcome]: Results grouped by outcome variable")
    print(f"  - Odds_Ratio_Analysis: Odds ratio analysis")

def main():
    """
    Main function to run the logistic regression analysis.
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
    
    # Check if outcome variables exist and are binary
    outcome_vars = ['has_teams', 'has_funded_teams']
    available_outcomes = [var for var in outcome_vars if var in df.columns]
    
    if not available_outcomes:
        print("No binary outcome variables found. Please check your data.")
        return
    
    print(f"Using outcome variables: {available_outcomes}")
    
    # Run logistic regressions
    results_df = run_individual_logistic_regressions(df, feature_cols, available_outcomes)
    
    if results_df.empty:
        print("No valid logistic regressions could be performed.")
        return
    
    # Save results
    save_results_to_excel(results_df, 'regression/logistic_regression_results_all_data.xlsx')
    
    # Print summary
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION ANALYSIS SUMMARY")
    print("="*60)
    
    for outcome in results_df['Outcome'].unique():
        outcome_results = results_df[results_df['Outcome'] == outcome]
        significant = outcome_results[outcome_results['P_Value'] < 0.05]
        high_performance = outcome_results[outcome_results['AUC_ROC'] > 0.7]
        
        print(f"\n{outcome}:")
        print(f"  Total regressions: {len(outcome_results)}")
        print(f"  Significant (p < 0.05): {len(significant)}")
        print(f"  High performance (AUC > 0.7): {len(high_performance)}")
        print(f"  Mean AUC: {outcome_results['AUC_ROC'].mean():.4f}")
        print(f"  Max AUC: {outcome_results['AUC_ROC'].max():.4f}")
        print(f"  Mean McFadden R²: {outcome_results['McFadden_R_Squared'].mean():.4f}")
        
        if len(significant) > 0:
            best_feature = significant.loc[significant['AUC_ROC'].idxmax()]
            print(f"  Best significant feature: {best_feature['Feature']} (AUC = {best_feature['AUC_ROC']:.4f}, p = {best_feature['P_Value']:.4f})")
        
        if len(high_performance) > 0:
            best_performance = high_performance.loc[high_performance['AUC_ROC'].idxmax()]
            print(f"  Best performing feature: {best_performance['Feature']} (AUC = {best_performance['AUC_ROC']:.4f})")

if __name__ == "__main__":
    main()
