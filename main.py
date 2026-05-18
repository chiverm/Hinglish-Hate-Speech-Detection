1# DEPRECATED: This file is retained for reference only.
# Use train.py for all experiments and training runs.
# Example: python train.py --model baseline
#          python train.py --model transformer --epochs 3
#          python train.py --model all

import numpy as np
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing.load_data import load_and_preprocess_dataset
from src.models.baseline import train_baseline, evaluate_model as eval_baseline
from src.models.transformer import load_model_and_tokenizer, train_transformer, evaluate_transformer
from src.evaluation.metrics import compare_models, generate_metrics_report


def create_output_directories():
    """Create necessary output directories."""
    dirs = [
        project_root / 'models',
        project_root / 'results',
        project_root / 'logs'
    ]
    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True)
    print("✓ Output directories created/verified\n")


def load_and_clean_data(dataset_path: str, text_column: str, label_column: str):
    """
    Load and clean dataset.
    
    Args:
        dataset_path (str): Path to CSV file
        text_column (str): Name of text column
        label_column (str): Name of label column
        
    Returns:
        Tuple of train/test splits
    """
    print("=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80 + "\n")
    
    # Load and preprocess dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_dataset(
        filepath=dataset_path,
        text_column=text_column,
        label_column=label_column,
        test_size=0.2,
        random_state=42
    )
    
    # Additional preprocessing: clean text (optional but recommended)
    print("\nCleaning text data...")
    print("  - Removing URLs, mentions, hashtags")
    print("  - Converting to lowercase")
    print("  - Handling extra whitespace\n")
    
    X_train_clean = np.array([
        __import__('src.preprocessing.clean_text', fromlist=['clean_text']).clean_text(text)
        for text in X_train
    ])
    X_test_clean = np.array([
        __import__('src.preprocessing.clean_text', fromlist=['clean_text']).clean_text(text)
        for text in X_test
    ])
    
    print(f"✓ Data loading and cleaning complete")
    print(f"  X_train_clean[0]: {X_train_clean[0][:80]}...\n")
    
    return X_train_clean, X_test_clean, y_train, y_test


def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train baseline model (TF-IDF + Logistic Regression).
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Tuple of (model, vectorizer, metrics)
    """
    print("=" * 80)
    print("STEP 2: TRAINING BASELINE MODEL")
    print("=" * 80 + "\n")
    
    # Train baseline
    model, vectorizer = train_baseline(X_train, y_train)
    
    # Evaluate
    metrics = eval_baseline(model, vectorizer, X_test, y_test, plot_confusion_matrix=True)
    
    return model, vectorizer, metrics


def train_transformer_model(X_train, y_train, X_test, y_test):
    """
    Train transformer model (XLM-RoBERTa).
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Tuple of (model, tokenizer, metrics)
    """
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING TRANSFORMER MODEL")
    print("=" * 80 + "\n")
    
    # Split training data for validation
    split_idx = int(0.8 * len(X_train))
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer('xlm-roberta-base')
    
    # Train transformer
    model, tokenizer = train_transformer(
        X_train_split, y_train_split,
        X_val, y_val,
        tokenizer, model,
        output_dir=str(project_root / 'models' / 'transformer_model'),
        num_epochs=2,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = evaluate_transformer(model, tokenizer, X_test, y_test)
    
    return model, tokenizer, metrics


def evaluate_and_compare(baseline_model, baseline_vectorizer, baseline_metrics,
                         transformer_model, transformer_tokenizer, transformer_metrics):
    """
    Comprehensive evaluation and comparison of models.
    
    Args:
        baseline_model, baseline_vectorizer: Baseline model components
        baseline_metrics: Baseline metrics
        transformer_model, transformer_tokenizer: Transformer components
        transformer_metrics: Transformer metrics
    """
    print("\n" + "=" * 80)
    print("STEP 4: MODEL EVALUATION AND COMPARISON")
    print("=" * 80 + "\n")
    
    # Compare models
    compare_models({
        'Baseline (TF-IDF + LR)': baseline_metrics,
        'Transformer (XLM-RoBERTa)': transformer_metrics
    })


def save_models(baseline_model, baseline_vectorizer, transformer_model, transformer_tokenizer):
    """
    Save trained models and tokenizers.
    
    Args:
        baseline_model: Baseline model
        baseline_vectorizer: TF-IDF vectorizer
        transformer_model: Transformer model
        transformer_tokenizer: Tokenizer
    """
    print("\n" + "=" * 80)
    print("STEP 5: SAVING MODELS")
    print("=" * 80 + "\n")
    
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save baseline model
    print("Saving baseline model...")
    if baseline_model is not None:
        baseline_model_path = models_dir / 'baseline_model.pkl'
        baseline_vectorizer_path = models_dir / 'baseline_vectorizer.pkl'
        
        with open(baseline_model_path, 'wb') as f:
            pickle.dump(baseline_model, f)
        with open(baseline_vectorizer_path, 'wb') as f:
            pickle.dump(baseline_vectorizer, f)
        
        print(f"  ✓ Baseline model saved to: {baseline_model_path}")
        print(f"  ✓ Baseline vectorizer saved to: {baseline_vectorizer_path}\n")
    else:
        print("  ⚠ Baseline model is None, skipping...\n")
    
    # Save transformer model
    print("Saving transformer model...")
    if transformer_model is not None:
        transformer_model_path = models_dir / 'transformer_model'
        transformer_model.save_pretrained(str(transformer_model_path))
        transformer_tokenizer.save_pretrained(str(transformer_model_path))
        
        print(f"  ✓ Transformer model and tokenizer saved to: {transformer_model_path}\n")
    else:
        print("  ⚠ Transformer model is None, skipping...\n")


def generate_reports(baseline_metrics, transformer_metrics, X_test, y_test_baseline, y_test_transformer):
    """
    Generate evaluation reports.
    
    Args:
        baseline_metrics: Baseline model metrics
        transformer_metrics: Transformer model metrics
        X_test: Test texts
        y_test_baseline: Baseline test labels
        y_test_transformer: Transformer test labels
    """
    print("=" * 80)
    print("STEP 6: GENERATING REPORTS")
    print("=" * 80 + "\n")
    
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Generate baseline report
    baseline_report_path = results_dir / 'baseline_evaluation_report.txt'
    baseline_report = generate_metrics_report(
        baseline_metrics,
        'Baseline Model (TF-IDF + Logistic Regression)',
        str(baseline_report_path)
    )
    print("✓ Baseline evaluation report generated\n")
    
    # Generate transformer report
    transformer_report_path = results_dir / 'transformer_evaluation_report.txt'
    transformer_report = generate_metrics_report(
        transformer_metrics,
        'Transformer Model (XLM-RoBERTa)',
        str(transformer_report_path)
    )
    print("✓ Transformer evaluation report generated\n")
    
    # Generate summary report
    summary_report_path = results_dir / 'model_comparison_summary.txt'
    with open(summary_report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HATE SPEECH DETECTION MODEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASELINE MODEL (TF-IDF + LOGISTIC REGRESSION)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {baseline_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {baseline_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {baseline_metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {baseline_metrics['f1']:.4f}\n\n")
        
        f.write("TRANSFORMER MODEL (XLM-RoBERTa)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {transformer_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {transformer_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {transformer_metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {transformer_metrics['f1']:.4f}\n\n")
        
        # Calculate differences
        acc_diff = transformer_metrics['accuracy'] - baseline_metrics['accuracy']
        f1_diff = transformer_metrics['f1'] - baseline_metrics['f1']
        
        f.write("PERFORMANCE DIFFERENCE (Transformer - Baseline)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy Change:  {acc_diff:+.4f} ({acc_diff/baseline_metrics['accuracy']*100:+.2f}%)\n")
        f.write(f"F1 Score Change:  {f1_diff:+.4f} ({f1_diff/baseline_metrics['f1']*100:+.2f}%)\n\n")
        
        if transformer_metrics['f1'] > baseline_metrics['f1']:
            f.write("✓ TRANSFORMER MODEL PERFORMS BETTER\n")
        else:
            f.write("✓ BASELINE MODEL PERFORMS BETTER\n")
    
    print(f"✓ Model comparison summary saved to: {summary_report_path}\n")


def main():
    """Main pipeline execution."""
    
    print("\n" + "=" * 80)
    print("HATE SPEECH DETECTION MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directories
    create_output_directories()
    
    # Define dataset path
    dataset_path = str(project_root / 'data' / 'raw' / 'combined_hate_speech_dataset.csv')
    
    # Check if dataset exists, if not create sample data
    if not os.path.exists(dataset_path):
        print(f"⚠ Dataset not found at {dataset_path}")
        print("Creating sample dataset for demonstration...\n")
        
        # Create sample dataset
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        import pandas as pd
        
        sample_data = {
            'text': [
                "I love this movie", "You are so stupid", "Great day today",
                "I hate you all", "This is amazing", "Go die", "Beautiful sunset",
                "Seriously hate this", "Wonderful experience", "You should die",
                "I enjoy this", "Pathetic performance", "Amazing work", "You are disgusting",
                "Have a nice day", "I despise you"
            ],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv(dataset_path, index=False)
        print(f"✓ Sample dataset created at: {dataset_path}\n")
    
    # Load and clean data
    X_train, X_test, y_train, y_test = load_and_clean_data(
        dataset_path,
        text_column='text',
        label_column='hate_label'
    )
    
    # Train baseline model
    try:
        baseline_model, baseline_vectorizer, baseline_metrics = train_baseline_model(
            X_train, y_train, X_test, y_test
        )
        baseline_success = True
    except Exception as e:
        print(f"✗ Error training baseline model: {str(e)}")
        baseline_success = False
    
    # Train transformer model
    try:
        transformer_model, transformer_tokenizer, transformer_metrics = train_transformer_model(
            X_train, y_train, X_test, y_test
        )
        transformer_success = True
    except Exception as e:
        print(f"✗ Error training transformer model: {str(e)}")
        transformer_success = False
    
    # Evaluate and compare
    if baseline_success and transformer_success:
        evaluate_and_compare(
            baseline_model, baseline_vectorizer, baseline_metrics,
            transformer_model, transformer_tokenizer, transformer_metrics
        )
        
        # Save models
        save_models(
            baseline_model, baseline_vectorizer,
            transformer_model, transformer_tokenizer
        )
        
        # Generate reports
        generate_reports(
            baseline_metrics, transformer_metrics,
            X_test, y_test, y_test
        )
    elif baseline_success:
        print("\n✓ Baseline model training successful")
        save_models(baseline_model, baseline_vectorizer, None, None)
    elif transformer_success:
        print("\n✓ Transformer model training successful")
        save_models(None, None, transformer_model, transformer_tokenizer)
    
    # Final summary
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Next Steps:")
    print("1. Review evaluation reports in ./results/")
    print("2. Check confusion matrices in ./results/")
    print("3. Use trained models for predictions with app/app.py")
    print("4. Run SHAP analysis for model explainability")
    print("5. Fine-tune hyperparameters based on results\n")


if __name__ == "__main__":
    main()
