# Quick Start Guide: Regression Analyses

## 🎯 Choose Your Analysis

### **Question 1: What predicts the NUMBER of teams formed?**
```bash
python negative_binomial_regression_analysis.py
```
**Best for**: Count outcomes (0, 1, 2, 3, ...) with overdispersion

### **Question 2: What predicts WHETHER teams form (yes/no)?**
```bash
python logistic_regression_analysis.py
```
**Best for**: Binary outcomes (0 or 1)

### **Question 3: Do EARLY features predict LATER features?**
```bash
python temporal_analysis.py
```
**Best for**: Within-session prediction (beginning → middle → end)

### **Question 4: What are continuous relationships?**
```bash
python linear_regression_analysis.py
```
**Best for**: Continuous normally-distributed outcomes

---

## 📊 Your Data Summary

Based on your files in `data/{conf_name}/featurized_with_when/`:

### **Each Session Has 3 Segments:**
- `_beginning.json` - First 1/3 of meeting
- `_middle.json` - Middle 1/3 of meeting  
- `_end.json` - Last 1/3 of meeting

### **Example Features:**
- `num_evaluation_practices` - Count of evaluation behaviors
- `num_people_evaluation_practices` - Number of people participating
- `mean_score_evaluation_practices` - Average quality score

---

## 🚀 Running Your First Analysis

### **Step 1: Run Temporal Analysis (Recommended Start)**
```bash
cd /Users/eveyhuang/Documents/NICO/gemini_code/regression
python temporal_analysis.py
```

This will:
- ✓ Load all temporal data (beginning, middle, end)
- ✓ Test if beginning features predict middle features
- ✓ Test if beginning features predict end features
- ✓ Test if beginning predicts CHANGE over time
- ✓ Save 4 Excel files with results

### **Step 2: Check Results**
Open these files in Excel:
- `temporal_beginning_to_middle_results.xlsx`
- `temporal_beginning_to_end_results.xlsx`
- `temporal_middle_to_end_results.xlsx`
- `temporal_change_score_results.xlsx`

### **Step 3: Interpret**
Look for:
- **P_Value < 0.05**: Statistically significant
- **R_Squared > 0.1**: Meaningful effect
- **Coefficient**: Effect size (standardized if normalized)

---

## 📈 Example Output

After running temporal analysis, you might see:

```
=== OUTCOME VARIABLE DISTRIBUTION ANALYSIS ===
Dataset shape: (157, 45)

--- num_evaluation_practices ---
Beginning → End: R² = 0.45, β = 0.67, p < 0.001
Interpretation: Strong temporal stability - sessions starting with 
high evaluation maintain it through the end
```

---

## 🔍 What Each Analysis Tells You

### **Temporal Analysis**
**Question**: "If a session starts with high evaluation, does it stay high?"
- **High R² (0.5+)**: Strong stability/momentum
- **Low R² (<0.2)**: Flexible, changeable
- **Negative β**: Regression to mean / ceiling effect

### **Negative Binomial**
**Question**: "What increases the number of teams formed?"
- **IRR > 1**: Feature increases team count
- **IRR < 1**: Feature decreases team count
- **Example**: IRR = 1.5 means 50% more teams

### **Logistic Regression**
**Question**: "What makes teams more likely to form?"
- **OR > 1**: Feature increases odds
- **OR < 1**: Feature decreases odds
- **Example**: OR = 2.5 means 2.5x higher odds

---

## 🎓 Interpretation Guide

### **R² (R-Squared)**
- **0.0 - 0.1**: Weak relationship
- **0.1 - 0.3**: Moderate relationship
- **0.3 - 0.5**: Strong relationship
- **0.5+**: Very strong relationship

### **P-Value**
- **< 0.001**: *** Highly significant
- **< 0.01**: ** Very significant
- **< 0.05**: * Significant
- **≥ 0.05**: Not significant

### **Standardized Coefficient (β)**
- **0.0 - 0.1**: Negligible effect
- **0.1 - 0.3**: Small effect
- **0.3 - 0.5**: Medium effect
- **0.5+**: Large effect

---

## 💡 Common Findings & Interpretations

### **Finding 1: High Temporal Stability**
```
Beginning → End: R² = 0.60, β = 0.75, p < 0.001
```
**Means**: Early patterns strongly predict later patterns
**Implication**: First impressions matter; early intervention crucial

### **Finding 2: Ceiling Effect**
```
Beginning → Change: R² = 0.25, β = -0.50, p < 0.01
```
**Means**: High starters show less growth
**Implication**: Diminishing returns at high levels

### **Finding 3: Momentum Effect**
```
Beginning → Change: R² = 0.30, β = 0.55, p < 0.01
```
**Means**: High starters show even more growth
**Implication**: "Rich get richer" - early success breeds more success

### **Finding 4: No Temporal Relationship**
```
Beginning → End: R² = 0.05, β = 0.15, p = 0.20
```
**Means**: Beginning doesn't predict end; flexible trajectory
**Implication**: Middle segment matters; interventions can work

---

## 🔧 Customization

### **Change Control Variables**
Edit the script (around line 520):
```python
control_vars = ['num_facilitator', 'total_utterances', 'YOUR_VARIABLE']
```

### **Focus on Specific Features**
Comment out features you don't want to analyze.

### **Change Significance Level**
Use p < 0.01 instead of p < 0.05 for more stringent testing.

---

## ⚠️ Common Issues

### **Issue 1: "No temporal data found"**
**Solution**: Check that `data/` directory exists and contains subdirectories with `featurized_with_when/` folders

### **Issue 2: "Insufficient data"**
**Solution**: Need at least 10 sessions with complete data (beginning, middle, end)

### **Issue 3: Missing control variables**
**Solution**: Check column names match exactly (case-sensitive)

---

## 📝 Next Steps

1. **Run temporal analysis** to see overall patterns
2. **Identify significant predictors** (p < 0.05, R² > 0.1)
3. **Examine specific features** that are most predictive
4. **Consider mechanisms** - why do certain features predict?
5. **Run outcome analyses** (negative binomial) to link features to team formation

---

## 📚 More Information

- **Full guide**: `temporal_analysis_guide.md`
- **Model comparison**: `README_all_analyses.md`
- **Negative binomial**: `negative_binomial_guide.md`

---

## 🎉 Quick Win

**Want to see results in 2 minutes?**

```bash
cd /Users/eveyhuang/Documents/NICO/gemini_code/regression
python temporal_analysis.py
```

Then open `temporal_beginning_to_end_results.xlsx` and look at the **Significant_Results** sheet!

This shows which beginning features significantly predict end features. Sort by R_Squared to see the strongest predictors.






