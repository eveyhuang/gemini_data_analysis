#!/usr/bin/env python3
"""
ML Analysis Script for Binary Classification
============================================
Runs Linear Regression (LPM), Lasso Logistic Regression, and Random Forest
with 20 random 80/20 train-test splits. Visualizes best results.

Usage:
    python ml_analysis.py <outcome_variable>
    python ml_analysis.py has_funded_teams
    python ml_analysis.py has_teams
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, r2_score)
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


def load_data(data_path='data/all_data_df.xlsx'):
    """Load and prepare data."""
    df = pd.read_excel(data_path)
    print(f"✅ Loaded data with shape: {df.shape}")
    return df


def prepare_features(df, outcome):
    """Prepare features and target variable."""
    # Exclude outcome variables and identifiers
    exclude_cols = ['num_teams', 'num_funded_teams', 'conference_name', 'session_id', 
                    'has_teams', 'has_funded_teams', 'conference', 'session', 'segment']
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[outcome].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Remove rows where target is missing
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y, feature_cols


def run_lasso_logistic(X, y, feature_cols, n_splits=20):
    """Run Lasso Logistic Regression with multiple splits."""
    print("\n" + "="*60)
    print("🔵 LASSO LOGISTIC REGRESSION")
    print("="*60)
    
    results = []
    np.random.seed(42)
    seeds = np.random.randint(1, 100000, n_splits)
    
    for i, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(
            penalty='l1', solver='saga', C=1.0, max_iter=5000,
            random_state=42, class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        results.append({
            'split': i+1, 'seed': seed, 'accuracy': acc, 'f1': f1, 'auc': auc,
            'confusion_matrix': cm, 'fpr': fpr, 'tpr': tpr, 'y_test': y_test,
            'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'model': model, 'scaler': scaler
        })
        
        if (i + 1) % 5 == 0:
            print(f"   Completed {i+1}/{n_splits} splits...")
    
    return results


def run_random_forest(X, y, feature_cols, n_splits=20):
    """Run Random Forest with multiple splits."""
    print("\n" + "="*60)
    print("🌲 RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    results = []
    np.random.seed(42)
    seeds = np.random.randint(1, 100000, n_splits)
    
    for i, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        results.append({
            'split': i+1, 'seed': seed, 'accuracy': acc, 'f1': f1, 'auc': auc,
            'confusion_matrix': cm, 'fpr': fpr, 'tpr': tpr, 'y_test': y_test,
            'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'model': model, 'scaler': scaler
        })
        
        if (i + 1) % 5 == 0:
            print(f"   Completed {i+1}/{n_splits} splits...")
    
    return results


def run_linear_regression(X, y, feature_cols, n_splits=20):
    """Run Linear Regression (LPM) with multiple splits."""
    print("\n" + "="*60)
    print("📈 LINEAR REGRESSION (Linear Probability Model)")
    print("="*60)
    
    results = []
    np.random.seed(42)
    seeds = np.random.randint(1, 100000, n_splits)
    
    for i, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Get predictions and clip for probability interpretation
        y_pred_proba = model.predict(X_test_scaled)
        y_pred_proba_clipped = np.clip(y_pred_proba, 0, 1)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba_clipped)
        r2 = r2_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_clipped)
        
        results.append({
            'split': i+1, 'seed': seed, 'accuracy': acc, 'f1': f1, 'auc': auc, 'r2': r2,
            'confusion_matrix': cm, 'fpr': fpr, 'tpr': tpr, 'y_test': y_test,
            'y_pred': y_pred, 'y_pred_proba': y_pred_proba_clipped, 'model': model, 'scaler': scaler
        })
        
        if (i + 1) % 5 == 0:
            print(f"   Completed {i+1}/{n_splits} splits...")
    
    return results


def print_summary(results, model_name):
    """Print summary statistics for all splits."""
    print(f"\n{'Split':<8} {'Seed':<8} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['split']:<8} {r['seed']:<8} {r['accuracy']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}")
    print("-"*60)
    
    mean_acc = np.mean([r['accuracy'] for r in results])
    mean_f1 = np.mean([r['f1'] for r in results])
    mean_auc = np.mean([r['auc'] for r in results])
    std_acc = np.std([r['accuracy'] for r in results])
    std_f1 = np.std([r['f1'] for r in results])
    std_auc = np.std([r['auc'] for r in results])
    
    print(f"{'Mean':<8} {'':<8} {mean_acc:<12.4f} {mean_f1:<12.4f} {mean_auc:<12.4f}")
    print(f"{'Std':<8} {'':<8} {std_acc:<12.4f} {std_f1:<12.4f} {std_auc:<12.4f}")
    
    best_idx = np.argmax([r['auc'] for r in results])
    best = results[best_idx]
    print(f"\n🏆 BEST SPLIT: Split {best['split']} (seed={best['seed']})")
    print(f"   Accuracy: {best['accuracy']:.4f} | F1: {best['f1']:.4f} | AUC: {best['auc']:.4f}")
    
    return best


def visualize_lasso(results, feature_cols, outcome, output_path):
    """Visualize Lasso Logistic Regression results."""
    best = results[np.argmax([r['auc'] for r in results])]
    
    # Calculate baselines
    y_test = best['y_test']
    pos_rate = y_test.mean()
    baseline_accuracy = max(pos_rate, 1 - pos_rate)
    baseline_f1 = 2 * pos_rate / (1 + pos_rate) if pos_rate > 0 else 0
    baseline_auc = 0.5
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Performance Metrics with Baselines
    ax1 = axes[0, 0]
    metrics_names = ['Accuracy', 'F1 Score', 'AUC-ROC']
    values = [best['accuracy'], best['f1'], best['auc']]
    baselines = [baseline_accuracy, baseline_f1, baseline_auc]
    colors = ['#00C853', '#2979FF', '#FF6D00']
    
    bars = ax1.bar(metrics_names, values, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Performance Metrics', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # for i, baseline in enumerate(baselines):
    #     ax1.hlines(y=baseline, xmin=i-0.35, xmax=i+0.35, colors='red', linestyles='--', linewidth=2.5)
    #     ax1.scatter([i], [baseline], color='red', s=80, marker='D', zorder=5)
    #     ax1.text(i + 0.38, baseline, f'{baseline:.2f}', va='center', ha='left', fontsize=9, color='red', fontweight='bold')
    
    # legend_elements = [Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, marker='D', markersize=8, label='Random Baseline')]
    # ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: ROC Curve
    ax2 = axes[0, 1]
    ax2.plot(best['fpr'], best['tpr'], color='#2979FF', linewidth=2.5, label=f'Model (AUC = {best["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Random (AUC = 0.5)')
    ax2.fill_between(best['fpr'], best['tpr'], alpha=0.2, color='#2979FF')
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Confusion Matrix
    ax3 = axes[1, 0]
    cm = best['confusion_matrix']
    im = ax3.imshow(cm, cmap='Blues')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Negative', 'Positive'], fontsize=10)
    ax3.set_yticklabels(['Negative', 'Positive'], fontsize=10)
    ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax3.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)
    
    # Plot 4: Top 10 Features
    ax4 = axes[1, 1]
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': best['model'].coef_[0],
        'abs_coefficient': np.abs(best['model'].coef_[0])
    })
    nonzero = coef_df[coef_df['abs_coefficient'] > 0].copy()
    top_features = nonzero.nlargest(10, 'abs_coefficient').sort_values('coefficient')
    top_features['clean_name'] = top_features['feature'].apply(lambda x: x.replace('_', ' ').capitalize())
    
    y_pos = np.arange(len(top_features))
    bar_colors = ['#00C853' if c > 0 else '#FF1744' for c in top_features['coefficient']]
    ax4.barh(y_pos, top_features['coefficient'], color=bar_colors, edgecolor='white', linewidth=1.2)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_features['clean_name'], fontsize=11)
    ax4.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
    ax4.set_title('Top 10 Features', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.suptitle(f'Lasso Logistic Regression: {outcome} (Best Split {best["split"]})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {output_path}")


def visualize_random_forest(results, feature_cols, outcome, output_path):
    """Visualize Random Forest results."""
    best = results[np.argmax([r['auc'] for r in results])]
    
    # Calculate baselines
    y_test = best['y_test']
    pos_rate = y_test.mean()
    baseline_accuracy = max(pos_rate, 1 - pos_rate)
    baseline_f1 = 2 * pos_rate / (1 + pos_rate) if pos_rate > 0 else 0
    baseline_auc = 0.5
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Performance Metrics with Baselines
    ax1 = axes[0, 0]
    metrics_names = ['Accuracy', 'F1 Score', 'AUC-ROC']
    values = [best['accuracy'], best['f1'], best['auc']]
    baselines = [baseline_accuracy, baseline_f1, baseline_auc]
    colors = ['#00C853', '#2979FF', '#FF6D00']
    
    bars = ax1.bar(metrics_names, values, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Performance Metrics', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # for i, baseline in enumerate(baselines):
    #     ax1.hlines(y=baseline, xmin=i-0.35, xmax=i+0.35, colors='red', linestyles='--', linewidth=2.5)
    #     ax1.scatter([i], [baseline], color='red', s=80, marker='D', zorder=5)
    #     ax1.text(i + 0.38, baseline, f'{baseline:.2f}', va='center', ha='left', fontsize=9, color='red', fontweight='bold')
    
    # legend_elements = [Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, marker='D', markersize=8, label='Random Baseline')]
    # ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: ROC Curve
    ax2 = axes[0, 1]
    ax2.plot(best['fpr'], best['tpr'], color='#FF6D00', linewidth=2.5, label=f'Model (AUC = {best["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Random (AUC = 0.5)')
    ax2.fill_between(best['fpr'], best['tpr'], alpha=0.2, color='#FF6D00')
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Confusion Matrix
    ax3 = axes[1, 0]
    cm = best['confusion_matrix']
    im = ax3.imshow(cm, cmap='Oranges')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Negative', 'Positive'], fontsize=10)
    ax3.set_yticklabels(['Negative', 'Positive'], fontsize=10)
    ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax3.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)
    
    # Plot 4: Top 10 Features (Gini Importance)
    ax4 = axes[1, 1]
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': best['model'].feature_importances_
    })
    top_features = imp_df.nlargest(10, 'importance').sort_values('importance')
    top_features['clean_name'] = top_features['feature'].apply(lambda x: x.replace('_', ' ').capitalize())
    
    y_pos = np.arange(len(top_features))
    bars = ax4.barh(y_pos, top_features['importance'], color='#FF6D00', edgecolor='white', linewidth=1.2)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_features['clean_name'], fontsize=11)
    ax4.set_xlabel('Feature Importance (Gini)', fontsize=12, fontweight='bold')
    ax4.set_title('Top 10 Features', fontsize=14, fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    for bar in bars:
        ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Random Forest: {outcome} (Best Split {best["split"]})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {output_path}")


def visualize_linear_regression(results, feature_cols, outcome, output_path):
    """Visualize Linear Regression (LPM) results."""
    best = results[np.argmax([r['auc'] for r in results])]
    
    # Calculate baselines
    y_test = best['y_test']
    pos_rate = y_test.mean()
    baseline_accuracy = max(pos_rate, 1 - pos_rate)
    baseline_f1 = 2 * pos_rate / (1 + pos_rate) if pos_rate > 0 else 0
    baseline_auc = 0.5
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Performance Metrics with Baselines
    ax1 = axes[0, 0]
    metrics_names = ['Accuracy', 'F1 Score', 'AUC-ROC', 'R²']
    values = [best['accuracy'], best['f1'], best['auc'], max(0, best['r2'])]
    # baselines = [baseline_accuracy, baseline_f1, baseline_auc, 0]
    colors = ['#00C853', '#2979FF', '#FF6D00', '#9C27B0']
    
    bars = ax1.bar(metrics_names, values, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Performance Metrics', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # for i, baseline in enumerate(baselines):
    #     ax1.hlines(y=baseline, xmin=i-0.35, xmax=i+0.35, colors='red', linestyles='--', linewidth=2.5)
    #     ax1.scatter([i], [baseline], color='red', s=80, marker='D', zorder=5)
    #     ax1.text(i + 0.38, baseline, f'{baseline:.2f}', va='center', ha='left', fontsize=9, color='red', fontweight='bold')
    
    # legend_elements = [Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, marker='D', markersize=8, label='Random Baseline')]
    # ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: ROC Curve
    ax2 = axes[0, 1]
    ax2.plot(best['fpr'], best['tpr'], color='#9C27B0', linewidth=2.5, label=f'Model (AUC = {best["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Random (AUC = 0.5)')
    ax2.fill_between(best['fpr'], best['tpr'], alpha=0.2, color='#9C27B0')
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Confusion Matrix
    ax3 = axes[1, 0]
    cm = best['confusion_matrix']
    im = ax3.imshow(cm, cmap='Purples')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Negative', 'Positive'], fontsize=10)
    ax3.set_yticklabels(['Negative', 'Positive'], fontsize=10)
    ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax3.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)
    
    # Plot 4: Top 10 Features (Coefficients)
    ax4 = axes[1, 1]
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': best['model'].coef_,
        'abs_coefficient': np.abs(best['model'].coef_)
    })
    top_features = coef_df.nlargest(10, 'abs_coefficient').sort_values('coefficient')
    top_features['clean_name'] = top_features['feature'].apply(lambda x: x.replace('_', ' ').capitalize())
    
    y_pos = np.arange(len(top_features))
    bar_colors = ['#00C853' if c > 0 else '#FF1744' for c in top_features['coefficient']]
    ax4.barh(y_pos, top_features['coefficient'], color=bar_colors, edgecolor='white', linewidth=1.2)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_features['clean_name'], fontsize=11)
    ax4.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
    ax4.set_title('Top 10 Features', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.suptitle(f'Linear Regression (LPM): {outcome} (Best Split {best["split"]})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ML models (Linear Regression, Lasso Logistic, Random Forest) on binary outcome.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ml_analysis.py has_funded_teams
    python ml_analysis.py has_teams
    python ml_analysis.py has_funded_teams --splits 50
    python ml_analysis.py has_teams --data path/to/data.xlsx
        """
    )
    parser.add_argument("outcome", type=str, help="Outcome variable (e.g., has_funded_teams, has_teams)")
    parser.add_argument("--splits", type=int, default=20, help="Number of random train-test splits (default: 20)")
    parser.add_argument("--data", type=str, default="data/all_data_df.xlsx", help="Path to data file")
    args = parser.parse_args()
    
    outcome = args.outcome
    n_splits = args.splits
    data_path = args.data
    basename = Path(data_path).stem

    print("="*80)
    print(f"🚀 ML ANALYSIS: {outcome}")
    print(f"   Running {n_splits} random 80/20 train-test splits")
    print("="*80)
    
    # Create output directory
    output_dir = f"ml_viz/{outcome}/{basename}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load data
    df = load_data(data_path)
    
    # Check if outcome exists
    if outcome not in df.columns:
        print(f"❌ Error: Outcome '{outcome}' not found in data!")
        print(f"   Available columns: {df.columns.tolist()}")
        return
    
    # Prepare features
    X, y, feature_cols = prepare_features(df, outcome)
    print(f"\n📊 Data prepared:")
    print(f"   Samples: {len(y)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Class balance: {y.mean():.2%} positive")
    
    # Run models
    print("\n" + "="*80)
    print("🔬 RUNNING MODELS")
    print("="*80)
    
    # 1. Lasso Logistic Regression
    lasso_results = run_lasso_logistic(X, y, feature_cols, n_splits)
    best_lasso = print_summary(lasso_results, "Lasso Logistic")
    visualize_lasso(lasso_results, feature_cols, outcome, f"{output_dir}/lasso_logistic.png")
    
    # 2. Random Forest
    rf_results = run_random_forest(X, y, feature_cols, n_splits)
    best_rf = print_summary(rf_results, "Random Forest")
    visualize_random_forest(rf_results, feature_cols, outcome, f"{output_dir}/random_forest.png")
    
    # 3. Linear Regression (LPM)
    lr_results = run_linear_regression(X, y, feature_cols, n_splits)
    best_lr = print_summary(lr_results, "Linear Regression")
    visualize_linear_regression(lr_results, feature_cols, outcome, f"{output_dir}/linear_regression.png")
    
    # Final Summary
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON (Best Split AUC-ROC)")
    print("="*80)
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
    print("-"*60)
    print(f"{'Lasso Logistic':<25} {best_lasso['accuracy']:<12.4f} {best_lasso['f1']:<12.4f} {best_lasso['auc']:<12.4f}")
    print(f"{'Random Forest':<25} {best_rf['accuracy']:<12.4f} {best_rf['f1']:<12.4f} {best_rf['auc']:<12.4f}")
    print(f"{'Linear Regression':<25} {best_lr['accuracy']:<12.4f} {best_lr['f1']:<12.4f} {best_lr['auc']:<12.4f}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print("   - lasso_logistic.png")
    print("   - random_forest.png")
    print("   - linear_regression.png")


if __name__ == "__main__":
    main()

