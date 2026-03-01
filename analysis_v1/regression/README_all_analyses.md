# Complete Regression Analysis Suite

## Overview
This directory contains three complementary regression analysis approaches for your meeting data.

## Analysis Scripts

### **1. Linear Regression** (`linear_regression_analysis.py`)
- **For**: Continuous outcomes (if normally distributed)
- **Outcomes**: `num_teams`, `num_funded_teams` (if treating as continuous)
- **Best when**: Small variance/mean ratio, normally distributed

### **2. Logistic Regression** (`logistic_regression_analysis.py`)
- **For**: Binary outcomes
- **Outcomes**: `has_teams`, `has_funded_teams`
- **Best when**: Outcome is 0/1 (presence/absence)

### **3. Negative Binomial Regression** (`negative_binomial_regression_analysis.py`)
- **For**: Count outcomes with overdispersion
- **Outcomes**: `num_teams`, `num_funded_teams`
- **Best when**: Variance > Mean (overdispersion)

### **4. Temporal Analysis** (`temporal_analysis.py`)
- **For**: Predicting later segments from earlier segments
- **Questions**: Does beginning predict middle/end?
- **Best when**: You have temporal data (beginning, middle, end)

## Quick Comparison

| Analysis Type | Outcome Type | Key Metric | Interpretation |
|--------------|--------------|------------|----------------|
| **Linear** | Continuous | R² | Variance explained |
| **Logistic** | Binary (0/1) | AUC-ROC, Odds Ratio | Classification accuracy |
| **Negative Binomial** | Count with overdispersion | IRR, McFadden R² | Multiplicative effect |
| **Temporal** | Any (within-session) | R², Coefficient | Predictive power over time |

## Decision Tree: Which Analysis to Use?

```
What is your research question?
│
├─ Predicting team formation COUNTS (0, 1, 2, 3, ...)
│  │
│  ├─ Is variance > mean? (overdispersion)
│  │  ├─ YES → Negative Binomial Regression ✓
│  │  └─ NO → Linear Regression or Poisson
│  │
│  └─ Want to treat as continuous?
│     └─ Linear Regression (less ideal for counts)
│
├─ Predicting team formation PRESENCE (yes/no, 0/1)
│  └─ Logistic Regression ✓
│
└─ Predicting LATER segments from EARLY segments
   └─ Temporal Analysis ✓
```

## Your Data Characteristics

Based on your outcome variables:

### **`num_teams` and `num_funded_teams`**
- **Type**: Count data (0, 1, 2, 3, ...)
- **Expected**: Likely overdispersed (variance > mean)
- **Recommended**: **Negative Binomial Regression**
- **Alternative**: Linear regression (if treating as continuous)

### **`has_teams` and `has_funded_teams`**
- **Type**: Binary (0 = no teams, 1 = teams formed)
- **Recommended**: **Logistic Regression**

### **Temporal Features (beginning → middle → end)**
- **Type**: Within-session prediction
- **Recommended**: **Temporal Analysis**

## Common Features Across All Scripts

### **✓ Control Variables**
All scripts support control variables:
```python
control_vars = ['meeting_length', 'num_members', 'num_facilitator', 'total_utterances']
```

### **✓ Data Normalization**
All scripts normalize features (except temporal change scores):
```python
normalize=True  # Standardizes all features
```

### **✓ Comprehensive Output**
All scripts produce Excel files with:
- All results
- Significant results (p < 0.05)
- High performance results
- Results grouped by outcome

### **✓ Statistical Rigor**
All scripts calculate:
- Coefficients with standard errors
- Confidence intervals (95%)
- P-values
- Model fit metrics

## How to Run Each Analysis

### **1. For Count Outcomes (Recommended)**
```bash
cd /Users/eveyhuang/Documents/NICO/gemini_code/regression
python negative_binomial_regression_analysis.py
```
**Output**: `negative_binomial_regression_results.xlsx`

### **2. For Binary Outcomes**
```bash
python logistic_regression_analysis.py
```
**Output**: `logistic_regression_results_all_data.xlsx`

### **3. For Linear Relationships**
```bash
python linear_regression_analysis.py
```
**Output**: `linear_regression_results_with_controls.xlsx`

### **4. For Temporal Prediction**
```bash
python temporal_analysis.py
```
**Output**: Multiple files (beginning_to_middle, beginning_to_end, etc.)

## Installation

### **Required Packages**
```bash
pip install pandas numpy scipy scikit-learn openpyxl statsmodels
```

Or use the requirements file:
```bash
pip install -r requirements_negative_binomial.txt
```

## Output Interpretation

### **Linear Regression**
```
R² = 0.35, β = 0.45, p < 0.001
→ Feature explains 35% of variance
→ 1 SD increase in feature → 0.45 SD increase in outcome
```

### **Logistic Regression**
```
AUC = 0.75, OR = 2.5, p < 0.01
→ 75% classification accuracy
→ Feature increases odds of team formation by 2.5x
```

### **Negative Binomial**
```
McFadden R² = 0.20, IRR = 1.5, p < 0.001
→ Good model fit (R² > 0.1)
→ Feature increases expected count by 50%
```

### **Temporal Analysis**
```
Beginning → End: R² = 0.60, β = 0.75, p < 0.001
→ Beginning strongly predicts end (60% variance)
→ High temporal stability
```

## Recommended Workflow

### **Step 1: Exploratory Analysis**
```bash
python negative_binomial_regression_analysis.py
```
- Checks overdispersion automatically
- Provides model recommendations
- Shows distribution characteristics

### **Step 2: Main Analysis**
Run the appropriate model based on Step 1 recommendations:
- **Negative Binomial** for count outcomes with overdispersion
- **Logistic** for binary outcomes
- **Linear** for continuous outcomes

### **Step 3: Temporal Analysis**
```bash
python temporal_analysis.py
```
- Examines within-session dynamics
- Tests predictive power of early features
- Identifies temporal patterns

### **Step 4: Compare Results**
- Check consistency across models
- Look for robust predictors (significant in multiple analyses)
- Interpret effect sizes in context

## Key Features Analyzed

All scripts analyze these feature categories:

### **Behavioral Counts**
- `num_evaluation_practices`
- `num_information_seeking`
- `num_knowledge_sharing`
- `num_idea_management`
- `num_relational_climate`
- `num_coordination_decision`
- `num_integration_practices`

### **Participation Breadth**
- `num_people_evaluation_practices`
- `num_people_information_seeking`
- etc.

### **Quality Scores**
- `mean_score_evaluation_practices`
- `mean_score_information_seeking`
- `mean_score_overall`

### **Aggregate Metrics**
- `total_utterances`
- `total_score`
- `positive_intensity`
- `num_interruption`

## Best Practices

### **1. Choose the Right Model**
- Use the decision tree above
- Check distribution assumptions
- Consider research question

### **2. Include Control Variables**
```python
control_vars = [
    'meeting_length',    # Session duration
    'num_members',       # Group size
    'num_facilitator',   # Leadership
    'total_utterances'   # Activity level
]
```

### **3. Normalize Features**
- Always use `normalize=True` for fair comparison
- Standardized coefficients are interpretable
- Helps with model convergence

### **4. Check Assumptions**
- **Linear**: Normal residuals, homoscedasticity
- **Logistic**: No perfect separation
- **Negative Binomial**: Overdispersion present
- **Temporal**: Sufficient sample size (n > 10)

### **5. Interpret Effect Sizes**
- Don't rely only on p-values
- Check R², AUC, or McFadden R²
- Consider practical significance

## Troubleshooting

### **"Singular matrix" Error**
- Multicollinearity issue
- Remove highly correlated features
- Check VIF (Variance Inflation Factor)

### **"Convergence failed" Warning**
- Increase `maxiter` parameter
- Check for extreme outliers
- Consider feature scaling

### **"Insufficient data" Message**
- Need more observations
- Reduce number of control variables
- Check for missing values

### **Low R² / Poor Fit**
- May indicate complex relationships
- Consider interaction terms
- Check for non-linear patterns

## Getting Help

### **Documentation**
- `linear_regression_analysis.py` - Standard linear models
- `logistic_regression_analysis.py` - Binary outcomes
- `negative_binomial_regression_analysis.py` - Count data
- `temporal_analysis.py` - Time-based prediction

### **Guides**
- `regression_model_guide.md` - General model selection
- `negative_binomial_guide.md` - NB-specific guidance
- `temporal_analysis_guide.md` - Temporal analysis details

## Summary

**For your specific research questions:**

1. **"What predicts team formation (count)?"**
   → Use Negative Binomial Regression

2. **"What predicts team formation (yes/no)?"**
   → Use Logistic Regression

3. **"Do early features predict later features?"**
   → Use Temporal Analysis

4. **"What's the relationship with continuous outcomes?"**
   → Use Linear Regression

All scripts are ready to run and will automatically:
- Load your data
- Handle missing values
- Include control variables
- Normalize features
- Calculate comprehensive statistics
- Save results to Excel

Good luck with your analyses!






