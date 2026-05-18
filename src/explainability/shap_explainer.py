import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
from typing import Any, Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class BasetlineModelExplainer:
    """
    SHAP explainer for baseline models (TF-IDF + Logistic Regression).
    """
    
    def __init__(self, model: Any, vectorizer: Any, background_texts: np.ndarray = None):
        """
        Initialize explainer for baseline model.
        
        Args:
            model: Trained LogisticRegression model
            vectorizer: Fitted TfidfVectorizer
            background_texts (np.ndarray, optional): Background texts for SHAP
        """
        self.model = model
        self.vectorizer = vectorizer
        self.background_texts = background_texts
        self.explainer = None
        
    def _predict_fn(self, texts_list: List[str]) -> np.ndarray:
        """
        Prediction function for SHAP.
        
        Args:
            texts_list (List[str]): List of texts
            
        Returns:
            np.ndarray: Prediction probabilities for class 1 (hate)
        """
        X_tfidf = self.vectorizer.transform(texts_list)
        return self.model.predict_proba(X_tfidf)[:, 1]
    
    def setup_explainer(self, num_background: int = 100):
        """
        Setup SHAP explainer with background texts.
        
        Args:
            num_background (int): Number of background samples. Default: 100
        """
        print("Setting up SHAP explainer for baseline model...")
        
        if self.background_texts is None or len(self.background_texts) == 0:
            print("  Creating synthetic background texts...")
            # Create simple background texts
            self.background_texts = np.array([
                "I love this", "This is good", "Bad experience", "Not okay",
                "Great job", "Awesome work", "Terrible thing", "Very bad",
                "Fantastic", "Wonderful", "Horrible", "Awful"
            ])
        
        # Use subset of background texts for efficiency
        background_subset = self.background_texts[:min(num_background, len(self.background_texts))]
        X_background = self.vectorizer.transform(background_subset)
        
        # Create explainer using KernelExplainer (works with any model)
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba,
            X_background
        )
        print(f"  ✓ SHAP explainer initialized with {len(background_subset)} background samples\n")
    
    def explain_prediction(self, text: str, plot: bool = True) -> Dict[str, Any]:
        """
        Explain prediction for a single text using SHAP values.
        
        Args:
            text (str): Text to explain
            plot (bool): Whether to plot explanation. Default: True
            
        Returns:
            Dict with explanation data
        """
        if self.explainer is None:
            self.setup_explainer()
        
        # Vectorize input
        X_text = self.vectorizer.transform([text])
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_text)
        
        # Get prediction
        prediction = self.model.predict([text])[0]
        prediction_proba = self.model.predict_proba(X_text)[0]
        
        # Get feature names and their SHAP values
        feature_names = self.vectorizer.get_feature_names_out()
        feature_indices = X_text.nonzero()[1]  # Get indices of non-zero features
        
        # Create explanation dictionary
        explanation = {
            'text': text,
            'prediction': 'HATE' if prediction == 1 else 'NON-HATE',
            'probability_non_hate': prediction_proba[0],
            'probability_hate': prediction_proba[1],
            'shap_values': shap_values,
            'feature_names': feature_names,
            'feature_indices': feature_indices,
            'important_features': []
        }
        
        # Extract important features
        if isinstance(shap_values, list):
            # For binary classification
            shap_vals = shap_values[1]  # Get values for class 1 (hate)
        else:
            shap_vals = shap_values
        
        # Get feature contributions
        feature_contributions = []
        for idx in feature_indices:
            feature_contributions.append({
                'feature': feature_names[idx],
                'shap_value': shap_vals[0, idx] if len(shap_vals.shape) > 1 else shap_vals[idx],
                'value': X_text[0, idx]
            })
        
        # Sort by absolute SHAP value
        feature_contributions = sorted(feature_contributions, 
                                       key=lambda x: abs(x['shap_value']), 
                                       reverse=True)
        explanation['important_features'] = feature_contributions[:10]  # Top 10
        
        # Print explanation
        self._print_explanation(explanation)
        
        # Plot if requested
        if plot:
            self._plot_explanation(explanation, text)
        
        return explanation
    
    def _print_explanation(self, explanation: Dict):
        """Print text-based explanation."""
        print("\n" + "=" * 80)
        print("SHAP EXPLANATION - BASELINE MODEL (TF-IDF + Logistic Regression)")
        print("=" * 80 + "\n")
        
        print(f"Text: \"{explanation['text']}\"")
        print(f"\nPrediction: {explanation['prediction']}")
        print(f"  ├─ Probability (Non-Hate): {explanation['probability_non_hate']:.4f}")
        print(f"  └─ Probability (Hate):     {explanation['probability_hate']:.4f}\n")
        
        print("Top Contributing Words (by SHAP value):\n")
        print(f"{'Rank':<6} {'Word':<20} {'SHAP Value':<15} {'Impact':<15}")
        print("-" * 56)
        
        for i, feature in enumerate(explanation['important_features'], 1):
            impact = "↑ Positive (Hate)" if feature['shap_value'] > 0 else "↓ Negative (Non-Hate)"
            print(f"{i:<6} {feature['feature']:<20} {feature['shap_value']:<15.6f} {impact:<15}")
    
    def _plot_explanation(self, explanation: Dict, text: str):
        """Plot SHAP explanation."""
        features = explanation['important_features']
        names = [f['feature'] for f in features]
        values = [f['shap_value'] for f in features]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]
        bars = ax.barh(names, values, color=colors, alpha=0.7)
        
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features (Words)', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Explanation: "{text[:50]}..."\n' + 
                    f'Prediction: {explanation["prediction"]} (Hate Prob: {explanation["probability_hate"]:.4f})',
                    fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', alpha=0.7, label='Increases Hate Prediction'),
            Patch(facecolor='#1f77b4', alpha=0.7, label='Decreases Hate Prediction')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('shap_explanation_baseline.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ SHAP explanation plot saved to: shap_explanation_baseline.png")
        plt.close()


class TransformerModelExplainer:
    """
    SHAP explainer for transformer models.
    """
    
    def __init__(self, model: Any, tokenizer: Any, device: str = 'auto'):
        """
        Initialize explainer for transformer model.
        
        Args:
            model: Trained transformer model
            tokenizer: Tokenizer
            device (str): Device to use. Default: 'auto'
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.explainer = None
    
    def _predict_fn(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for SHAP.
        
        Args:
            texts (List[str]): List of texts
            
        Returns:
            np.ndarray: Prediction probabilities for class 1
        """
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs[0, 1].item())  # Probability of class 1
        
        return np.array(predictions)
    
    def explain_prediction(self, text: str, plot: bool = True) -> Dict[str, Any]:
        """
        Explain prediction for transformer model using attention-based explanation.
        
        Args:
            text (str): Text to explain
            plot (bool): Whether to plot explanation. Default: True
            
        Returns:
            Dict with explanation data
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction and attention
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            attentions = outputs.attentions
        
        prediction = torch.argmax(probs, dim=1).item()
        prob_hate = probs[0, 1].item()
        prob_non_hate = probs[0, 0].item()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Compute attention-based importance
        # Average attention across all layers and heads
        attention_weights = []
        for layer_attention in attentions:
            # Layer attention shape: (batch, num_heads, seq_len, seq_len)
            layer_avg = layer_attention[0].mean(dim=0)  # Average over heads
            attention_weights.append(layer_avg)
        
        # Compute token importance from attention
        all_attention = torch.stack(attention_weights).mean(dim=0)  # Average over layers
        token_importance = all_attention[-1, 1:-1].cpu().numpy()  # Last token attending to other tokens
        
        # Create explanation dictionary
        explanation = {
            'text': text,
            'prediction': 'HATE' if prediction == 1 else 'NON-HATE',
            'probability_non_hate': prob_non_hate,
            'probability_hate': prob_hate,
            'tokens': tokens,
            'token_importance': token_importance,
            'important_tokens': []
        }
        
        # Extract important tokens
        important_indices = np.argsort(-token_importance)[:10]  # Top 10
        for idx in important_indices:
            if idx < len(tokens) and tokens[idx] not in ['[CLS]', '[SEP]', '[PAD]']:
                explanation['important_tokens'].append({
                    'token': tokens[idx],
                    'importance': token_importance[idx]
                })
        
        # Print explanation
        self._print_explanation(explanation)
        
        # Plot if requested
        if plot:
            self._plot_explanation(explanation, text)
        
        return explanation
    
    def _print_explanation(self, explanation: Dict):
        """Print text-based explanation."""
        print("\n" + "=" * 80)
        print("SHAP EXPLANATION - TRANSFORMER MODEL")
        print("=" * 80 + "\n")
        
        print(f"Text: \"{explanation['text']}\"")
        print(f"\nPrediction: {explanation['prediction']}")
        print(f"  ├─ Probability (Non-Hate): {explanation['probability_non_hate']:.4f}")
        print(f"  └─ Probability (Hate):     {explanation['probability_hate']:.4f}\n")
        
        print("Top Important Tokens (by Attention):\n")
        print(f"{'Rank':<6} {'Token':<20} {'Importance':<15}")
        print("-" * 41)
        
        for i, token_info in enumerate(explanation['important_tokens'], 1):
            print(f"{i:<6} {token_info['token']:<20} {token_info['importance']:<15.6f}")
    
    def _plot_explanation(self, explanation: Dict, text: str):
        """Plot attention-based explanation."""
        tokens = [t for t in explanation['important_tokens']]
        names = [t['token'] for t in tokens]
        values = [t['importance'] for t in tokens]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(names, values, color='#2ca02c', alpha=0.7)
        
        ax.set_xlabel('Attention-based Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tokens', fontsize=12, fontweight='bold')
        ax.set_title(f'Attention-based Explanation: "{text[:50]}..."\n' + 
                    f'Prediction: {explanation["prediction"]} (Hate Prob: {explanation["probability_hate"]:.4f})',
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('shap_explanation_transformer.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Attention explanation plot saved to: shap_explanation_transformer.png")
        plt.close()


def explain_prediction(text: str, model: Any, vectorizer_or_tokenizer: Any = None,
                      model_type: str = 'auto', plot: bool = True) -> Dict[str, Any]:
    """
    Unified interface for explaining predictions from any model.
    
    Args:
        text (str): Text to explain
        model: Trained model
        vectorizer_or_tokenizer: TfidfVectorizer or Tokenizer
        model_type (str): 'baseline' or 'transformer' or 'auto'. Default: 'auto'
        plot (bool): Whether to plot explanation. Default: True
        
    Returns:
        Dict with explanation
        
    Example:
        >>> explanation = explain_prediction("I hate this", model, vectorizer, 'baseline')
    """
    # Auto-detect model type
    if model_type == 'auto':
        if hasattr(model, 'predict_proba'):
            model_type = 'baseline'
        elif hasattr(model, 'generate'):
            model_type = 'transformer'
        else:
            model_type = 'transformer'  # Default to transformer
    
    if model_type == 'baseline':
        explainer = BasetlineModelExplainer(model, vectorizer_or_tokenizer)
        return explainer.explain_prediction(text, plot=plot)
    else:
        explainer = TransformerModelExplainer(model, vectorizer_or_tokenizer)
        return explainer.explain_prediction(text, plot=plot)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SHAP EXPLAINABILITY DEMONSTRATION")
    print("=" * 80)
    
    print("\nThis module provides SHAP-based explainability for hate speech classifiers.")
    print("\nSupported Models:")
    print("  1. Baseline Model (TF-IDF + Logistic Regression)")
    print("  2. Transformer Models (XLM-RoBERTa, BERT, etc.)")
    
    print("\nUsage Examples:")
    print("\n1. For Baseline Model:")
    print("   from src.models.baseline import train_baseline")
    print("   from src.explainability.shap_explainer import explain_prediction")
    print("   ")
    print("   model, vectorizer = train_baseline(X_train, y_train)")
    print("   explanation = explain_prediction('I hate this', model, vectorizer, 'baseline')")
    
    print("\n2. For Transformer Model:")
    print("   from src.models.transformer import load_model_and_tokenizer, train_transformer")
    print("   from src.explainability.shap_explainer import explain_prediction")
    print("   ")
    print("   tokenizer, model = load_model_and_tokenizer()")
    print("   explanation = explain_prediction('I hate this', model, tokenizer, 'transformer')")
    
    print("\n" + "=" * 80)
    print("Features:")
    print("=" * 80)
    print("✓ SHAP values for baseline models (TF-IDF features)")
    print("✓ Attention-based explanations for transformers")
    print("✓ Visual plots of important features/tokens")
    print("✓ Probability and confidence scores")
    print("✓ Top-N important features/tokens extraction")
    print("✓ Support for both binary classification tasks")
