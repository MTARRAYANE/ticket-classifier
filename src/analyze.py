"""
Visualization and Analysis Module

Provides utilities for visualizing model performance and analyzing predictions.
"""

import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_metrics(metrics_path="../models/model_metrics.joblib"):
    """
    Load saved training metrics.
    
    Args:
        metrics_path (str): Path to metrics file
    
    Returns:
        list: Training results with metrics
    """
    try:
        return joblib.load(metrics_path)
    except FileNotFoundError:
        print(f"Metrics file not found at {metrics_path}")
        return None


def print_evaluation_report(results):
    """
    Print formatted evaluation report from training results.
    
    Args:
        results (list): List of (model_name, metrics) tuples
    """
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("MODEL EVALUATION REPORT")
    print("="*80)
    
    for model_name, metrics in results:
        print(f"\nModel: {model_name}")
        print("-" * 80)
        
        # Basic metrics
        print(f"Test Accuracy:              {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score:             {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score:          {metrics['weighted_f1']:.4f}")
        
        # Cross-validation metrics if available
        if "cv_mean" in metrics:
            print(f"Cross-Val F1 (mean):        {metrics['cv_mean']:.4f}")
            print(f"Cross-Val F1 (std):         {metrics['cv_std']:.4f}")
        
        # Per-class metrics
        if "classification_report" in metrics:
            print("\nPer-Class Performance:")
            report = metrics["classification_report"]
            for class_name, scores in report.items():
                if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                    print(f"  {class_name:20s} - Precision: {scores['precision']:.4f}, "
                          f"Recall: {scores['recall']:.4f}, F1: {scores['f1-score']:.4f}")


def plot_confusion_matrix(results, output_path="../models/confusion_matrix.png"):
    """
    Plot and save confusion matrix heatmap for best model.
    
    Args:
        results (list): Training results
        output_path (str): Path to save the figure
    """
    if not results:
        print("No results to plot")
        return
    
    # Use last model (typically best model)
    model_name, metrics = results[-1]
    cm = np.array(metrics.get("confusion_matrix", []))
    
    if cm.size == 0:
        print("No confusion matrix data available")
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()


def plot_metrics_comparison(results, output_path="../models/metrics_comparison.png"):
    """
    Plot comparison of different models' performance metrics.
    
    Args:
        results (list): Training results
        output_path (str): Path to save the figure
    """
    if not results:
        print("No results to plot")
        return
    
    model_names = [name for name, _ in results]
    accuracies = [metrics['accuracy'] for _, metrics in results]
    macro_f1s = [metrics['macro_f1'] for _, metrics in results]
    weighted_f1s = [metrics['weighted_f1'] for _, metrics in results]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
    ax.bar(x, macro_f1s, width, label='Macro F1', color='lightcoral')
    ax.bar(x + width, weighted_f1s, width, label='Weighted F1', color='lightgreen')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to {output_path}")
    plt.close()


def plot_class_performance(results, output_path="../models/class_performance.png"):
    """
    Plot per-class precision, recall, and F1 scores.
    
    Args:
        results (list): Training results
        output_path (str): Path to save the figure
    """
    if not results:
        print("No results to plot")
        return
    
    # Use best model (last one)
    model_name, metrics = results[-1]
    report = metrics.get("classification_report", {})
    
    if not report:
        print("No classification report available")
        return
    
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_name, scores in report.items():
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            classes.append(class_name)
            precisions.append(scores.get('precision', 0))
            recalls.append(scores.get('recall', 0))
            f1_scores.append(scores.get('f1-score', 0))
    
    if not classes:
        print("No per-class metrics found")
        return
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precisions, width, label='Precision', color='skyblue')
    ax.bar(x, recalls, width, label='Recall', color='lightcoral')
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Performance - {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class performance chart saved to {output_path}")
    plt.close()


def plot_cv_scores(results, output_path="../models/cv_scores.png"):
    """
    Plot cross-validation scores for each model.
    
    Args:
        results (list): Training results
        output_path (str): Path to save the figure
    """
    if not results:
        print("No results to plot")
        return
    
    model_names = [name for name, _ in results]
    cv_means = []
    cv_stds = []
    
    for _, metrics in results:
        cv_means.append(metrics.get('cv_mean', 0))
        cv_stds.append(metrics.get('cv_std', 0))
    
    if all(m == 0 for m in cv_means):
        print("No cross-validation data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(model_names))
    ax.bar(x, cv_means, yerr=cv_stds, capsize=5, color='skyblue', alpha=0.7, 
           error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Cross-Validation F1 Score')
    ax.set_title('Cross-Validation Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Cross-validation scores plot saved to {output_path}")
    plt.close()


def analyze_predictions(predictions_list):
    """
    Analyze a list of predictions to generate statistics.
    
    Args:
        predictions_list (list): List of prediction dictionaries
    
    Returns:
        dict: Analysis statistics
    """
    if not predictions_list:
        return {}
    
    confidences = [p["confidence"] for p in predictions_list]
    
    analysis = {
        "total_predictions": len(predictions_list),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "min_confidence": round(min(confidences), 4),
        "max_confidence": round(max(confidences), 4),
        "high_confidence_pct": round(
            sum(1 for c in confidences if c >= 0.8) / len(confidences) * 100, 2
        )
    }
    
    return analysis


def plot_confidence_distribution(predictions_list, output_path="../models/confidence_dist.png"):
    """
    Plot histogram of prediction confidence scores.
    
    Args:
        predictions_list (list): List of prediction dictionaries
        output_path (str): Path to save the figure
    """
    if not predictions_list:
        print("No predictions to plot")
        return
    
    confidences = [p["confidence"] for p in predictions_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
    ax.axvline(np.median(confidences), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(confidences):.3f}')
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Confidence Scores')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confidence distribution saved to {output_path}")
    plt.close()


def export_metrics_json(results, output_path="../models/metrics.json"):
    """
    Export metrics to JSON for easy sharing/visualization.
    
    Args:
        results (list): Training results
        output_path (str): Output file path
    """
    metrics_data = []
    
    for model_name, metrics in results:
        metrics_data.append({
            "model": model_name,
            "test_accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "cv_f1_mean": metrics.get("cv_mean"),
            "cv_f1_std": metrics.get("cv_std")
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✓ Metrics exported to {output_path}")


def generate_all_visualizations(results, output_dir="../models"):
    """
    Generate all available visualizations.
    
    Args:
        results (list): Training results
        output_dir (str): Directory to save visualizations
    """
    print("\nGenerating visualizations...")
    plot_metrics_comparison(results, f"{output_dir}/metrics_comparison.png")
    plot_confusion_matrix(results, f"{output_dir}/confusion_matrix.png")
    plot_class_performance(results, f"{output_dir}/class_performance.png")
    plot_cv_scores(results, f"{output_dir}/cv_scores.png")
    export_metrics_json(results, f"{output_dir}/metrics.json")
    print("✓ All visualizations generated!\n")


if __name__ == "__main__":
    """
    Example usage: Display training metrics and generate visualizations
    """
    results = load_metrics()
    if results:
        print_evaluation_report(results)
        generate_all_visualizations(results)
