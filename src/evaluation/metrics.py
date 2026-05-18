import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc, precision_recall_curve)
from typing import Dict, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of binary classification predictions.
    
    Computes all important metrics for hate speech classification:
    - Accuracy: Overall correctness
    - Precision: True positives among predicted positives (minimize false alarms)
    - Recall/Sensitivity: True positives among actual positives (catch all hate speech)
    - Specificity: True negatives among actual negatives
    - F1 Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under ROC curve (probability ranking quality)
    - Confusion Matrix: True/False Positives and Negatives
    
    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1)
        y_pred (np.ndarray): Predicted binary labels (0 or 1)
        y_pred_proba (np.ndarray, optional): Prediction probabilities for ROC-AUC.
                                             Shape: (n_samples, 2) or (n_samples,)
    
    Returns:
        Dict[str, Any]: Dictionary containing all computed metrics
        
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = evaluate_predictions(y_true, y_pred)
        >>> print(metrics['accuracy'], metrics['f1'])
    """
    print("\n" + "=" * 80)
    print("EVALUATING PREDICTIONS - COMPREHENSIVE METRICS")
    print("=" * 80 + "\n")
    
    # Basic metrics
    print("Step 1: Computing classification metrics\n")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics derived from confusion matrix
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Print basic metrics
    print(f"  CLASSIFICATION METRICS:")
    print(f"  ├─ Accuracy:              {accuracy:.4f}")
    print(f"  ├─ Precision:             {precision:.4f}")
    print(f"  ├─ Recall/Sensitivity:    {recall:.4f}")
    print(f"  ├─ Specificity:           {specificity:.4f}")
    print(f"  ├─ F1 Score:              {f1:.4f}\n")
    
    # ROC-AUC (if probabilities provided)
    roc_auc = None
    if y_pred_proba is not None:
        print("Step 2: Computing ROC-AUC score\n")
        
        # Handle different probability shapes
        if len(y_pred_proba.shape) == 2:
            proba_class_1 = y_pred_proba[:, 1]
        else:
            proba_class_1 = y_pred_proba
        
        try:
            roc_auc = roc_auc_score(y_true, proba_class_1)
            print(f"  ├─ ROC-AUC Score:         {roc_auc:.4f}\n")
        except Exception as e:
            print(f"  ├─ ROC-AUC Score:         Could not compute ({str(e)})\n")
    
    # Confusion matrix details
    print("Step 3: Confusion Matrix Analysis\n")
    print(f"  Confusion Matrix:")
    print(f"  ├─ True Negatives (TN):   {tn}")
    print(f"  ├─ False Positives (FP):  {fp}")
    print(f"  ├─ False Negatives (FN):  {fn}")
    print(f"  └─ True Positives (TP):   {tp}\n")
    
    # Error rates
    print(f"  Error Rates:")
    print(f"  ├─ False Positive Rate:   {false_positive_rate:.4f}")
    print(f"  └─ False Negative Rate:   {false_negative_rate:.4f}\n")
    
    # Classification report
    print("Step 4: Detailed Classification Report\n")
    print(classification_report(y_true, y_pred, 
                              target_names=['Non-Hate (0)', 'Hate (1)']))
    
    # Store all metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,  # Alias for recall
        'specificity': specificity,
        'f1': f1,
        'f1_score': f1,  # Alias
        'roc_auc': roc_auc,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         title: str = 'Confusion Matrix',
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (8, 6),
                         cmap: str = 'Blues',
                         show_percentages: bool = True) -> None:
    """
    Plot and display confusion matrix for binary classification.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        title (str): Plot title. Default: 'Confusion Matrix'
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size. Default: (8, 6)
        cmap (str): Colormap name. Default: 'Blues'
        show_percentages (bool): Show percentages instead of counts. Default: True
        
    Example:
        >>> plot_confusion_matrix(y_true, y_pred, 'My Model', 'cm.png')
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages if requested
    if show_percentages:
        cm_percent = cm.astype('float') / cm.sum() * 100
        annot = np.array([[f'{int(cm[i,j])}\n({cm_percent[i,j]:.1f}%)'
                          for j in range(cm.shape[1])]
                         for i in range(cm.shape[0])])
    else:
        annot = cm
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap,
                xticklabels=['Non-Hate (0)', 'Hate (1)'],
                yticklabels=['Non-Hate (0)', 'Hate (1)'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 12})
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  title: str = 'ROC Curve',
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Shows the trade-off between True Positive Rate and False Positive Rate
    across different classification thresholds.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred_proba (np.ndarray): Predicted probabilities
        title (str): Plot title. Default: 'ROC Curve'
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size. Default: (8, 6)
        
    Example:
        >>> plot_roc_curve(y_true, y_pred_proba, save_path='roc.png')
    """
    # Handle probability shape
    if len(y_pred_proba.shape) == 2:
        y_scores = y_pred_proba[:, 1]
    else:
        y_scores = y_pred_proba
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {save_path}")
    
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                               title: str = 'Precision-Recall Curve',
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot Precision-Recall curve.
    
    Shows the trade-off between precision and recall across different thresholds.
    Useful for imbalanced datasets.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred_proba (np.ndarray): Predicted probabilities
        title (str): Plot title. Default: 'Precision-Recall Curve'
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size. Default: (8, 6)
        
    Example:
        >>> plot_precision_recall_curve(y_true, y_pred_proba, 'pr_curve.png')
    """
    # Handle probability shape
    if len(y_pred_proba.shape) == 2:
        y_scores = y_pred_proba[:, 1]
    else:
        y_scores = y_pred_proba
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="best", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved to: {save_path}")
    
    plt.close()


def compare_models(models_metrics: Dict[str, Dict[str, Any]],
                  metrics_to_compare: list = None) -> None:
    """
    Compare metrics across multiple models.
    
    Args:
        models_metrics (Dict): Dictionary mapping model names to their metrics
        metrics_to_compare (list, optional): List of metric keys to compare.
                                            Default: ['accuracy', 'precision', 'recall', 'f1']
                                            
    Example:
        >>> baseline_metrics = evaluate_predictions(y_true_base, y_pred_base)
        >>> context_metrics = evaluate_predictions(y_true_ctx, y_pred_ctx)
        >>> compare_models({
        ...     'Baseline': baseline_metrics,
        ...     'Context-Aware': context_metrics
        ... })
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80 + "\n")
    
    # Find which metrics exist in all models
    available_metrics = []
    for metric in metrics_to_compare:
        if all(metric in m for m in models_metrics.values()):
            available_metrics.append(metric)
    
    # Build comparison table
    print(f"{'Model':<20}", end='')
    for metric in available_metrics:
        print(f"{metric:<15}", end='')
    print()
    print("-" * (20 + len(available_metrics) * 15))
    
    best_scores = {metric: -1 for metric in available_metrics}
    best_model = {metric: '' for metric in available_metrics}
    
    for model_name, metrics in models_metrics.items():
        print(f"{model_name:<20}", end='')
        for metric in available_metrics:
            value = metrics[metric]
            print(f"{value:<15.4f}", end='')
            
            if value > best_scores[metric]:
                best_scores[metric] = value
                best_model[metric] = model_name
        print()
    
    print("\n" + "=" * 80)
    print("BEST PERFORMING MODELS")
    print("=" * 80 + "\n")
    for metric in available_metrics:
        print(f"  {metric:<15}: {best_model[metric]:<20} ({best_scores[metric]:.4f})")


def generate_metrics_report(metrics: Dict[str, Any], model_name: str = 'Model',
                           save_path: Optional[str] = None) -> str:
    """
    Generate a text report of all metrics.
    
    Args:
        metrics (Dict): Metrics dictionary from evaluate_predictions()
        model_name (str): Name of the model. Default: 'Model'
        save_path (str, optional): Path to save the report
        
    Returns:
        str: Formatted report text
        
    Example:
        >>> report = generate_metrics_report(metrics, 'Baseline Model', 'report.txt')
    """
    report = []
    report.append("=" * 80)
    report.append(f"EVALUATION REPORT: {model_name}")
    report.append("=" * 80)
    report.append("")
    
    report.append("CLASSIFICATION METRICS:")
    report.append(f"  Accuracy:              {metrics['accuracy']:.4f}")
    report.append(f"  Precision:             {metrics['precision']:.4f}")
    report.append(f"  Recall (Sensitivity):  {metrics['recall']:.4f}")
    report.append(f"  Specificity:           {metrics['specificity']:.4f}")
    report.append(f"  F1 Score:              {metrics['f1']:.4f}")
    
    if metrics['roc_auc'] is not None:
        report.append(f"  ROC-AUC Score:         {metrics['roc_auc']:.4f}")
    
    report.append("")
    report.append("CONFUSION MATRIX:")
    report.append(f"  True Negatives (TN):   {metrics['true_negatives']}")
    report.append(f"  False Positives (FP):  {metrics['false_positives']}")
    report.append(f"  False Negatives (FN):  {metrics['false_negatives']}")
    report.append(f"  True Positives (TP):   {metrics['true_positives']}")
    
    report.append("")
    report.append("ERROR RATES:")
    report.append(f"  False Positive Rate:   {metrics['false_positive_rate']:.4f}")
    report.append(f"  False Negative Rate:   {metrics['false_negative_rate']:.4f}")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Report saved to: {save_path}")
    
    return report_text


if __name__ == "__main__":
    # Create sample predictions for demonstration
    print("\n" + "=" * 80)
    print("EVALUATION METRICS DEMONSTRATION")
    print("=" * 80)
    
    # Sample data
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    y_pred_proba = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4],
        [0.85, 0.15], [0.92, 0.08], [0.1, 0.9], [0.15, 0.85], [0.7, 0.3],
        [0.2, 0.8], [0.55, 0.45], [0.88, 0.12], [0.91, 0.09], [0.25, 0.75]
    ])
    
    # Evaluate predictions
    metrics = evaluate_predictions(y_true, y_pred, y_pred_proba)
    
    # Plot confusion matrix
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")
    
    plot_confusion_matrix(y_true, y_pred, 
                         'Confusion Matrix - Demo',
                         save_path='confusion_matrix_demo.png')
    
    plot_roc_curve(y_true, y_pred_proba,
                   'ROC Curve - Demo',
                   save_path='roc_curve_demo.png')
    
    plot_precision_recall_curve(y_true, y_pred_proba,
                               'Precision-Recall Curve - Demo',
                               save_path='pr_curve_demo.png')
    
    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80 + "\n")
    
    report = generate_metrics_report(metrics, 'Demo Model', 'evaluation_report.txt')
    print(report)
    
    # Compare models example
    print("\n" + "=" * 80)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 80 + "\n")
    
    y_pred_alt = np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
    y_pred_proba_alt = np.array([
        [0.95, 0.05], [0.1, 0.9], [0.7, 0.3], [0.85, 0.15], [0.3, 0.7],
        [0.88, 0.12], [0.91, 0.09], [0.12, 0.88], [0.2, 0.8], [0.75, 0.25],
        [0.15, 0.85], [0.2, 0.8], [0.89, 0.11], [0.93, 0.07], [0.2, 0.8]
    ])
    
    metrics_alt = evaluate_predictions(y_true, y_pred_alt, y_pred_proba_alt)
    
    compare_models({
        'Baseline Model': metrics,
        'Improved Model': metrics_alt
    })
