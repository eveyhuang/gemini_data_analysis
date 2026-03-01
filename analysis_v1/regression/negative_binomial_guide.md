# Negative Binomial Regression Analysis Guide

## Overview
This guide explains how to use the negative binomial regression script for count data with overdispersion, such as `num_teams` and `num_funded_teams`.

## When to Use Negative Binomial Regression

### **Perfect for Count Data with Overdispersion**
- **Count outcomes**: Non-negative integers (0, 1, 2, 3, ...)
- **Overdispersion**: Variance > Mean (common in count data)
- **Examples**: Number of teams formed, number of funded projects, number of participants

### **Key Indicators for Negative Binomial:**
1. **Count data**: Integer values ≥ 0
2. **Overdispersion**: Variance/Mean > 1
3. **Zero inflation**: Many zero values (optional, can use Zero-Inflated NB)

## Key Features of the Script

### **1. Automatic Overdispersion Detection**
```python
# The script automatically checks:
variance = df[outcome].var()
mean = df[outcome].mean()
overdispersion_ratio = variance / mean
is_overdispersed = variance > mean
```

### **2. Control Variables Support**
```python
control_vars = ['meeting_length', 'num_members', 'num_facilitator', 'total_utterances']
```

### **3. Data Normalization**
```python
# Features are standardized for fair comparison
normalize=True  # Default
```

## Key Output Metrics

### **Incidence Rate Ratio (IRR)**
- **Interpretation**: Multiplicative effect on the count
- **Example**: IRR = 1.5 means 50% increase in expected count
- **Formula**: IRR = exp(coefficient)

### **McFadden's R²**
- **Range**: 0 to 1 (closer to 1 = better fit)
- **Interpretation**: Proportion of variance explained
- **Good fit**: R² > 0.1

### **Alpha Parameter (Overdispersion)**
- **Interpretation**: Higher α = more overdispersion
- **Typical range**: 0.1 to 2.0
- **Significance**: α > 0 indicates overdispersion

## How to Run the Analysis

### **1. Install Dependencies**
```bash
pip install -r requirements_negative_binomial.txt
```

### **2. Run the Script**
```bash
cd /Users/eveyhuang/Documents/NICO/gemini_code/regression
python negative_binomial_regression_analysis.py
```

### **3. Customize Control Variables**
```python
# Edit the control_vars list in the script:
control_vars = ['meeting_length', 'num_members', 'num_facilitator', 'total_utterances']
```

## Interpreting Results

### **Coefficient Interpretation**
```python
# If coefficient = 0.5:
# IRR = exp(0.5) = 1.65
# This means: 65% increase in expected count for each unit increase in feature
```

### **Significance Testing**
- **P-value < 0.05**: Statistically significant
- **Z-statistic**: Test statistic for coefficient significance
- **95% Confidence Intervals**: Range of plausible values

### **Model Fit Assessment**
- **McFadden R²**: Overall model fit
- **AIC/BIC**: Model comparison (lower = better)
- **Alpha**: Overdispersion parameter

## Example Output

```
=== OUTCOME VARIABLE DISTRIBUTION ANALYSIS ===
Dataset shape: (157, 45)

--- num_teams ---
Data type: int64
Unique values: [0, 1, 2, 3, 4, 5, 6, 7]
Zero values: 45 (28.7%)
Variance: 2.15
Overdispersion ratio (variance/mean): 1.43
Overdispersed: True
Recommended model: Negative Binomial Regression
Reason: Count data with overdispersion (ratio: 1.43)
```

## Advantages Over Other Models

### **vs. Poisson Regression**
- **Poisson**: Assumes variance = mean
- **Negative Binomial**: Allows variance > mean (overdispersion)
- **Better fit**: For overdispersed count data

### **vs. Linear Regression**
- **Linear**: Assumes normal distribution
- **Negative Binomial**: Respects count data nature
- **No negative predictions**: Counts are always ≥ 0

### **vs. Logistic Regression**
- **Logistic**: Binary outcomes (0/1)
- **Negative Binomial**: Count outcomes (0, 1, 2, 3, ...)
- **More information**: Preserves count magnitude

## Model Selection Decision Tree

```
Is your outcome count data (integers ≥ 0)?
├─ NO → Use appropriate model (logistic, linear, etc.)
└─ YES → Is variance > mean (overdispersed)?
    ├─ NO → Poisson Regression
    └─ YES → Is zero inflation > 20%?
        ├─ YES → Zero-Inflated Negative Binomial
        └─ NO → Negative Binomial Regression
```

## Best Practices

### **1. Check Overdispersion First**
```python
# Before running, check:
variance = df['num_teams'].var()
mean = df['num_teams'].mean()
if variance > mean:
    print("Use Negative Binomial")
else:
    print("Consider Poisson")
```

### **2. Include Relevant Controls**
```python
# Good control variables:
control_vars = [
    'meeting_length',    # Session duration
    'num_members',      # Group size
    'num_facilitator',  # Leadership
    'total_utterances'  # Activity level
]
```

### **3. Interpret IRR Carefully**
- **IRR > 1**: Positive effect (increases count)
- **IRR < 1**: Negative effect (decreases count)
- **IRR = 1**: No effect
- **IRR = 2**: Doubles the expected count

### **4. Check Model Assumptions**
- **Overdispersion**: Alpha > 0
- **Model fit**: McFadden R² > 0.1
- **Significance**: P-value < 0.05

## Troubleshooting

### **Common Issues:**

1. **Convergence Problems**
   - Reduce number of features
   - Check for multicollinearity
   - Increase maxiter parameter

2. **Perfect Separation**
   - Some features perfectly predict outcome
   - Remove problematic features
   - Use regularization

3. **Insufficient Data**
   - Need at least 10 observations
   - More data needed with controls
   - Consider simpler models

### **Error Messages:**
- **"Singular matrix"**: Multicollinearity issue
- **"Convergence failed"**: Model fitting problem
- **"Insufficient data"**: Need more observations

## Next Steps

1. **Run the analysis** to see overdispersion detection
2. **Check results** for significant features
3. **Interpret IRR** for practical significance
4. **Compare models** using AIC/BIC
5. **Validate assumptions** with diagnostic plots

This negative binomial approach is perfect for your count data like `num_teams` and `num_funded_teams`!
