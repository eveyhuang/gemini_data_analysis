# Regression Model Selection Guide

## Overview
This guide helps you determine the appropriate regression model based on your outcome variable distributions.

## Key Changes Made to Logistic Regression Analysis

### 1. **Control Variables Support**
- Added `control_vars` parameter to include control variables in all models
- Automatically checks if control variables exist in the dataset
- Includes control variable information in results

### 2. **Data Normalization**
- Added `normalize` parameter (default: True)
- Uses `StandardScaler` to normalize features before regression
- Ensures all features are on the same scale for fair comparison

### 3. **Outcome Distribution Analysis**
- New `analyze_outcome_distributions()` function
- Automatically analyzes your outcome variables
- Provides model recommendations based on data characteristics

## Regression Model Selection Based on Outcome Variable Type

### **Binary Outcomes (0/1 values)**
- **Model**: Binary Logistic Regression
- **Example**: `has_teams`, `has_funded_teams`
- **Characteristics**: Only two possible values (0 and 1)
- **Use when**: Predicting presence/absence, success/failure

### **Count Data (Non-negative integers)**
- **Low zero inflation (<20% zeros)**:
  - **Model**: Poisson Regression
  - **Use when**: Count data with low zero inflation
- **High zero inflation (>20% zeros)**:
  - **Model**: Zero-Inflated Poisson or Negative Binomial
  - **Use when**: Many zero values (e.g., `num_teams` with many sessions having 0 teams)

### **Continuous Outcomes**
- **Model**: Linear Regression
- **Use when**: Continuous, normally distributed outcomes
- **Example**: Meeting duration, scores

### **Ordinal Categorical Outcomes**
- **Model**: Ordinal Logistic Regression
- **Use when**: Ordered categories (e.g., 1=low, 2=medium, 3=high)

## How to Use the Updated Code

### 1. **Basic Usage**
```python
# Run with default settings (normalization + control variables)
results_df = run_individual_logistic_regressions(
    df, 
    feature_cols, 
    outcome_vars=['has_teams', 'has_funded_teams'],
    control_vars=['meeting_length', 'num_members', 'num_facilitator'],
    normalize=True
)
```

### 2. **Customize Control Variables**
```python
# Define your own control variables
control_vars = ['meeting_length', 'num_members', 'num_facilitator', 'conference_year']
```

### 3. **Disable Normalization**
```python
# If you want to keep original scales
results_df = run_individual_logistic_regressions(
    df, feature_cols, outcome_vars, 
    control_vars=control_vars, 
    normalize=False
)
```

## Model Selection Decision Tree

```
Is your outcome variable binary (0/1)?
├─ YES → Binary Logistic Regression
└─ NO → Is it count data (non-negative integers)?
    ├─ YES → Is zero inflation > 20%?
    │   ├─ YES → Zero-Inflated Poisson/Negative Binomial
    │   └─ NO → Poisson/Negative Binomial Regression
    └─ NO → Is it ordinal categorical?
        ├─ YES → Ordinal Logistic Regression
        └─ NO → Linear Regression
```

## Benefits of the Updated Approach

### **1. Control Variables**
- **Isolates effects**: Controls for confounding variables
- **More accurate**: Separates main effect from control effects
- **Better interpretation**: Coefficients represent effect after controlling for other variables

### **2. Normalization**
- **Fair comparison**: All features on same scale
- **Better convergence**: Helps with model fitting
- **Interpretable coefficients**: Standardized effects

### **3. Automatic Analysis**
- **Data-driven decisions**: Based on actual data distributions
- **Model recommendations**: Suggests appropriate models
- **Validation**: Checks data quality and completeness

## Example Output

The analysis will show:
```
=== OUTCOME VARIABLE DISTRIBUTION ANALYSIS ===
Dataset shape: (157, 45)

--- num_teams ---
Data type: int64
Unique values: [0, 1, 2, 3, 4, 5, 6, 7]
Zero values: 45 (28.7%)
Recommended model: Zero-Inflated Poisson/Negative Binomial
Reason: Count data with high zero inflation (28.7% zeros)

--- has_teams ---
Data type: int64
Unique values: [0, 1]
Recommended model: Binary Logistic Regression
Reason: Binary outcome (0/1 values)
```

## Next Steps

1. **Run the analysis** to see your outcome variable distributions
2. **Choose appropriate models** based on the recommendations
3. **Customize control variables** based on your research questions
4. **Interpret results** considering the model type and controls used

This approach ensures you're using the most appropriate regression model for your specific data characteristics!
