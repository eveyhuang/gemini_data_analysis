# Temporal Predictive Analysis Guide

## Overview
This guide explains how to analyze whether features at the **beginning** of meetings predict features at **middle** and **end** segments.

## Research Questions Addressed

### **1. Cross-Lagged Prediction**
- **Question**: Do features at beginning predict features at middle/end?
- **Example**: Does high evaluation practice at beginning predict high evaluation at end?
- **Analysis**: Beginning → Middle, Beginning → End, Middle → End

### **2. Change Score Prediction**
- **Question**: Does beginning level predict the *change* over time?
- **Example**: Do sessions starting with low evaluation show greater increase?
- **Analysis**: Beginning value predicts (End - Beginning)

### **3. Temporal Stability**
- **Question**: How stable are features over time?
- **Measure**: Correlation between segments

## Data Structure

### **Input Data**
Your data is in: `data/{conf_name}/featurized_with_when/`

Each session has 3 files:
```
features_2020_11_05_NES_S1_beginning.json
features_2020_11_05_NES_S1_middle.json
features_2020_11_05_NES_S1_end.json
```

### **Aggregated Dataset**
The script creates a wide-format dataset:

| session | conf_name | num_members | beginning_num_evaluation | middle_num_evaluation | end_num_evaluation | ... |
|---------|-----------|-------------|--------------------------|----------------------|-------------------|-----|
| S1      | 2020NES   | 12          | 7                        | 6                    | 4                 | ... |
| S2      | 2020NES   | 10          | 5                        | 8                    | 6                 | ... |

## Analysis Types

### **1. Beginning → Middle Prediction**
```python
# Model: middle_feature = β₀ + β₁(beginning_feature) + controls
# Interpretation: How well does beginning predict middle?
```

**Use when**: You want to know if early patterns continue into middle segment

### **2. Beginning → End Prediction**
```python
# Model: end_feature = β₀ + β₁(beginning_feature) + controls
# Interpretation: Do early patterns persist until the end?
```

**Use when**: You want to know long-term predictive power

### **3. Middle → End Prediction**
```python
# Model: end_feature = β₀ + β₁(middle_feature) + controls
# Interpretation: Does middle predict end?
```

**Use when**: You want to know if mid-session patterns affect outcomes

### **4. Beginning → Change Prediction**
```python
# Model: (end - beginning) = β₀ + β₁(beginning_feature) + controls
# Interpretation: Does starting level predict growth?
```

**Use when**: You want to know if low/high starters show different trajectories

## How to Run

### **1. Run the Script**
```bash
cd /Users/eveyhuang/Documents/NICO/gemini_code/regression
python temporal_analysis.py
```

### **2. Output Files**
The script generates 4 Excel files:
- `temporal_beginning_to_middle_results.xlsx`
- `temporal_beginning_to_end_results.xlsx`
- `temporal_middle_to_end_results.xlsx`
- `temporal_change_score_results.xlsx`

Each file contains:
- **All_Results**: Complete results
- **Significant_Results**: p < 0.05
- **High_R_Squared**: R² > 0.1

## Interpreting Results

### **Coefficient Interpretation**

#### **For Direct Prediction (Beginning → End)**
```
Coefficient = 0.75, p < 0.001
Interpretation: A 1 SD increase in beginning evaluation predicts 
                 a 0.75 SD increase in end evaluation
```

#### **For Change Score Prediction**
```
Coefficient = -0.40, p < 0.01
Interpretation: Higher beginning values predict LESS change (ceiling effect)
                or
Coefficient = 0.40, p < 0.01  
Interpretation: Higher beginning values predict MORE change (momentum effect)
```

### **Key Metrics**

| Metric | Interpretation | Good Value |
|--------|----------------|------------|
| **R²** | Variance explained | > 0.1 |
| **P-value** | Statistical significance | < 0.05 |
| **Coefficient** | Effect size (standardized) | > 0.3 (medium) |
| **Correlation** | Simple relationship | > 0.3 |

## Example Findings

### **High Temporal Stability**
```
Beginning → End evaluation:
  R² = 0.65, β = 0.80, p < 0.001
```
**Interpretation**: Evaluation practices are highly stable; sessions that start with high evaluation maintain it.

### **Ceiling Effect**
```
Beginning → Change in evaluation:
  R² = 0.25, β = -0.50, p < 0.01
```
**Interpretation**: Sessions starting with high evaluation show less growth (ceiling effect).

### **Momentum Effect**
```
Beginning → Change in coordination:
  R² = 0.30, β = 0.55, p < 0.01
```
**Interpretation**: Sessions starting with coordination show more growth (momentum effect).

## Features to Analyze

### **Behavioral Counts**
- `num_evaluation_practices`
- `num_information_seeking`
- `num_knowledge_sharing`
- `num_idea_management`
- `num_relational_climate`
- `num_coordination_decision`
- `num_integration_practices`

### **Participation Metrics**
- `num_people_evaluation_practices`
- `num_people_information_seeking`
- `num_people_knowledge_sharing`
- etc.

### **Quality Metrics**
- `mean_score_evaluation_practices`
- `mean_score_information_seeking`
- `mean_score_knowledge_sharing`
- `mean_score_overall`
- `positive_intensity`

### **Aggregate Metrics**
- `total_utterances`
- `total_score`
- `num_interruption`

## Control Variables

The script automatically includes:
- **Session-level**: `num_members`, `meeting_length`
- **Segment-level**: `num_facilitator`, `total_utterances`

## Statistical Considerations

### **Autocorrelation**
- Features at beginning and end are from the same session
- Natural correlation expected
- Look for R² > 0.5 for strong prediction

### **Regression to the Mean**
- Extreme values at beginning tend toward average
- Negative coefficients in change models may indicate this

### **Sample Size**
- Need at least 10 complete sessions per conference
- More sessions = more reliable estimates

## Research Implications

### **If Beginning Strongly Predicts End:**
- Early intervention crucial
- First impressions matter
- Consider focusing on beginning segment

### **If Beginning Predicts Large Changes:**
- Momentum effects present
- Starting conditions matter
- Look for catalysts

### **If Beginning Doesn't Predict End:**
- Middle segment crucial
- Flexible trajectories
- Interventions can make a difference

## Next Steps

1. **Run the analysis**: `python temporal_analysis.py`
2. **Identify significant predictors**: Look at Significant_Results sheets
3. **Examine effect sizes**: Focus on R² > 0.1
4. **Interpret patterns**: Stability vs. change
5. **Consider mechanisms**: Why do certain features predict?

## Advanced Options

### **Customize Control Variables**
Edit the script (line ~520):
```python
control_vars = ['num_facilitator', 'total_utterances', 'your_variable']
```

### **Change Significance Threshold**
Edit the script to use p < 0.01 instead of p < 0.05

### **Add More Features**
The script automatically includes all features found in the JSON files

This temporal analysis will reveal whether early meeting dynamics predict later outcomes!






