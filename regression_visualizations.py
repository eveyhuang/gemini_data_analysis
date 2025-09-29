#!/usr/bin/env python3
"""
Regression Results Visualization Suite

This script creates comprehensive visualizations to compare regression results
across features and outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_regression_heatmap(results_df, metric='R_Squared', figsize=(12, 8)):
    """
    Create a heatmap showing regression results across features and outcomes.
    
    Args:
        results_df: DataFrame with regression results
        metric: Metric to visualize (R_Squared, P_Value, Coefficient, etc.)
        figsize: Figure size tuple
    """
    
    # Pivot the data for heatmap
    pivot_data = results_df.pivot(index='Feature', columns='Outcome', values=metric)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose colormap based on metric
    if metric == 'P_Value':
        cmap = 'RdYlBu_r'  # Red for low p-values (significant)
        vmin, vmax = 0, 1
    elif metric == 'R_Squared':
        cmap = 'viridis'
        vmin, vmax = 0, 1
    else:
        cmap = 'RdBu_r'
        vmin, vmax = None, None
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, 
                vmin=vmin, vmax=vmax, ax=ax, cbar_kws={'label': metric})
    
    ax.set_title(f'Regression {metric} Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Outcome Variables')
    ax.set_ylabel('Features')
    
    plt.tight_layout()
    return fig

def create_significance_matrix(results_df, figsize=(12, 8)):
    """
    Create a matrix showing significance levels across features and outcomes.
    """
    
    # Create significance matrix
    significance_matrix = results_df.pivot(index='Feature', columns='Outcome', values='P_Value')
    
    # Convert to significance levels
    sig_levels = significance_matrix.copy()
    sig_levels[sig_levels < 0.001] = 4  # ***
    sig_levels[(sig_levels >= 0.001) & (sig_levels < 0.01)] = 3  # **
    sig_levels[(sig_levels >= 0.01) & (sig_levels < 0.05)] = 2  # *
    sig_levels[(sig_levels >= 0.05) & (sig_levels < 0.1)] = 1  # .
    sig_levels[sig_levels >= 0.1] = 0  # ns
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap
    colors = ['white', 'lightcoral', 'orange', 'gold', 'lightgreen']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    im = ax.imshow(sig_levels.values, cmap=cmap, aspect='auto')
    
    # Add text annotations
    for i in range(len(sig_levels.index)):
        for j in range(len(sig_levels.columns)):
            p_val = significance_matrix.iloc[i, j]
            if p_val < 0.001:
                text = '***'
            elif p_val < 0.01:
                text = '**'
            elif p_val < 0.05:
                text = '*'
            elif p_val < 0.1:
                text = '.'
            else:
                text = 'ns'
            
            ax.text(j, i, text, ha='center', va='center', fontweight='bold')
    
    ax.set_xticks(range(len(sig_levels.columns)))
    ax.set_xticklabels(sig_levels.columns)
    ax.set_yticks(range(len(sig_levels.index)))
    ax.set_yticklabels(sig_levels.index)
    
    ax.set_title('Statistical Significance Matrix\n*** p<0.001, ** p<0.01, * p<0.05, . p<0.1, ns p≥0.1', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Outcome Variables')
    ax.set_ylabel('Features')
    
    plt.tight_layout()
    return fig

def create_coefficient_comparison(results_df, figsize=(15, 10)):
    """
    Create a comprehensive comparison of coefficients with confidence intervals.
    """
    
    # Separate by outcome
    outcomes = results_df['Outcome'].unique()
    n_outcomes = len(outcomes)
    
    fig, axes = plt.subplots(1, n_outcomes, figsize=figsize, sharey=True)
    if n_outcomes == 1:
        axes = [axes]
    
    for i, outcome in enumerate(outcomes):
        outcome_data = results_df[results_df['Outcome'] == outcome].copy()
        outcome_data = outcome_data.sort_values('Coefficient', ascending=True)
        
        # Create coefficient plot with confidence intervals
        y_pos = range(len(outcome_data))
        
        # Plot confidence intervals
        axes[i].errorbar(outcome_data['Coefficient'], y_pos, 
                       xerr=[outcome_data['Coefficient'] - outcome_data['CI_Lower_95'],
                             outcome_data['CI_Upper_95'] - outcome_data['Coefficient']],
                       fmt='o', capsize=5, capthick=2, markersize=8)
        
        # Color by significance
        colors = ['red' if p < 0.05 else 'gray' for p in outcome_data['P_Value']]
        for j, (coef, y) in enumerate(zip(outcome_data['Coefficient'], y_pos)):
            axes[i].scatter(coef, y, c=colors[j], s=100, zorder=5)
        
        # Add vertical line at zero
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(outcome_data['Feature'], fontsize=10)
        axes[i].set_xlabel('Coefficient Value')
        axes[i].set_title(f'{outcome}\nCoefficients with 95% CI')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Coefficient Comparison Across Features and Outcomes', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_r_squared_comparison(results_df, figsize=(12, 8)):
    """
    Create a bar chart comparing R-squared values across features and outcomes.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grouped bar chart
    x = np.arange(len(results_df['Feature'].unique()))
    width = 0.35
    
    outcomes = results_df['Outcome'].unique()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, outcome in enumerate(outcomes):
        outcome_data = results_df[results_df['Outcome'] == outcome]
        outcome_data = outcome_data.sort_values('Feature')
        
        bars = ax.bar(x + i * width, outcome_data['R_Squared'], width, 
                     label=outcome, color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, outcome_data['R_Squared']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('R-squared')
    ax.set_title('R-squared Comparison Across Features and Outcomes', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(results_df['Feature'].unique(), rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_p_value_distribution(results_df, figsize=(12, 6)):
    """
    Create visualizations of p-value distributions.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram of p-values
    ax1.hist(results_df['P_Value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of P-values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot for p-values (should be uniform under null)
    from scipy import stats
    stats.probplot(results_df['P_Value'], dist="uniform", plot=ax2)
    ax2.set_title('Q-Q Plot of P-values')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('P-value Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_feature_ranking(results_df, metric='R_Squared', figsize=(12, 8)):
    """
    Create a ranking visualization of features by performance metric.
    """
    
    # Calculate average performance by feature
    feature_performance = results_df.groupby('Feature')[metric].mean().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(feature_performance)), feature_performance.values, 
                   color='steelblue', alpha=0.8)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, feature_performance.values)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{value:.3f}', va='center', fontsize=10)
    
    ax.set_yticks(range(len(feature_performance)))
    ax.set_yticklabels(feature_performance.index)
    ax.set_xlabel(f'Average {metric}')
    ax.set_title(f'Feature Ranking by Average {metric}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_comprehensive_dashboard(results_df, save_path='regression_dashboard.png'):
    """
    Create a comprehensive dashboard with multiple visualizations.
    """
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. R-squared heatmap (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_data = results_df.pivot(index='Feature', columns='Outcome', values='R_Squared')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
    ax1.set_title('R-squared Heatmap')
    
    # 2. Significance matrix (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    sig_matrix = results_df.pivot(index='Feature', columns='Outcome', values='P_Value')
    sig_levels = sig_matrix.copy()
    sig_levels[sig_levels < 0.05] = 1
    sig_levels[sig_levels >= 0.05] = 0
    sns.heatmap(sig_levels, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
    ax2.set_title('Significance Matrix (p<0.05)')
    
    # 3. P-value distribution (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(results_df['P_Value'], bins=15, alpha=0.7, color='skyblue')
    ax3.axvline(x=0.05, color='red', linestyle='--', linewidth=2)
    ax3.set_title('P-value Distribution')
    ax3.set_xlabel('P-value')
    
    # 4. Coefficient comparison (middle row)
    ax4 = fig.add_subplot(gs[1, :])
    outcomes = results_df['Outcome'].unique()
    x = np.arange(len(results_df['Feature'].unique()))
    width = 0.35
    
    for i, outcome in enumerate(outcomes):
        outcome_data = results_df[results_df['Outcome'] == outcome]
        outcome_data = outcome_data.sort_values('Feature')
        ax4.bar(x + i * width, outcome_data['Coefficient'], width, 
               label=outcome, alpha=0.8)
    
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Coefficient Value')
    ax4.set_title('Coefficient Comparison')
    ax4.set_xticks(x + width/2)
    ax4.set_xticklabels(results_df['Feature'].unique(), rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature ranking (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    feature_avg = results_df.groupby('Feature')['R_Squared'].mean().sort_values()
    ax5.barh(range(len(feature_avg)), feature_avg.values, color='steelblue', alpha=0.8)
    ax5.set_yticks(range(len(feature_avg)))
    ax5.set_yticklabels(feature_avg.index)
    ax5.set_title('Feature Ranking (Avg R²)')
    
    # 6. R-squared comparison (bottom middle)
    ax6 = fig.add_subplot(gs[2, 1])
    for outcome in outcomes:
        outcome_data = results_df[results_df['Outcome'] == outcome]
        ax6.scatter(outcome_data['R_Squared'], outcome_data['P_Value'], 
                   label=outcome, alpha=0.7, s=100)
    
    ax6.axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
    ax6.set_xlabel('R-squared')
    ax6.set_ylabel('P-value')
    ax6.set_title('R² vs P-value')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Sample size analysis (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.scatter(results_df['N'], results_df['R_Squared'], 
               c=results_df['P_Value'], cmap='RdYlBu_r', alpha=0.7, s=100)
    ax7.set_xlabel('Sample Size (N)')
    ax7.set_ylabel('R-squared')
    ax7.set_title('Sample Size vs Performance')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Regression Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function to demonstrate all visualizations.
    """
    
    # Create sample data for demonstration
    print("Creating sample regression results...")
    np.random.seed(42)
    
    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    outcomes = ['num_teams', 'num_funded_teams']
    
    results = []
    for feature in features:
        for outcome in outcomes:
            # Simulate regression results
            r_squared = np.random.beta(2, 5)  # Skewed towards lower values
            p_value = np.random.beta(1, 3)    # Skewed towards lower values
            coefficient = np.random.normal(0, 0.5)
            
            results.append({
                'Feature': feature,
                'Outcome': outcome,
                'N': np.random.randint(50, 200),
                'Coefficient': coefficient,
                'Standard_Error': abs(np.random.normal(0.1, 0.05)),
                'P_Value': p_value,
                'R_Squared': r_squared,
                'Adjusted_R_Squared': r_squared * 0.9,
                'CI_Lower_95': coefficient - 1.96 * abs(np.random.normal(0.1, 0.05)),
                'CI_Upper_95': coefficient + 1.96 * abs(np.random.normal(0.1, 0.05))
            })
    
    results_df = pd.DataFrame(results)
    
    print("Creating visualizations...")
    
    # Create individual visualizations
    fig1 = create_regression_heatmap(results_df, 'R_Squared')
    fig2 = create_significance_matrix(results_df)
    fig3 = create_coefficient_comparison(results_df)
    fig4 = create_r_squared_comparison(results_df)
    fig5 = create_p_value_distribution(results_df)
    fig6 = create_feature_ranking(results_df)
    
    # Create comprehensive dashboard
    fig7 = create_comprehensive_dashboard(results_df)
    
    print("All visualizations created!")
    print("Individual plots can be saved by calling fig.savefig('filename.png')")
    
    return results_df

if __name__ == "__main__":
    results_df = main()
