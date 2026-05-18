import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Any, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class BaselineLimeExplainer:
    """
    LIME explainer for baseline models (TF-IDF + Logistic Regression).
    Uses LimeTextExplainer to explain individual predictions word-by-word.
    """

    def __init__(self, model: Any, vectorizer: Any, class_names: List[str] = None):
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names or ['Non-Hate', 'Hate']
        self._explainer = None

    def _predict_fn(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def _get_explainer(self):
        if self._explainer is None:
            from lime.lime_text import LimeTextExplainer
            self._explainer = LimeTextExplainer(class_names=self.class_names)
        return self._explainer

    def explain_prediction(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 500,
        plot: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using LIME.

        Args:
            text: Input text to explain.
            num_features: Number of top words to highlight.
            num_samples: Number of perturbed samples for LIME.
            plot: Whether to save a bar-chart PNG.

        Returns:
            dict with keys: text, prediction, probability_hate,
                            probability_non_hate, features, label
        """
        explainer = self._get_explainer()
        exp = explainer.explain_instance(
            text,
            self._predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)          # explain class 1 (Hate)
        )

        probs = self._predict_fn([text])[0]
        prediction = int(np.argmax(probs))
        label = 'HATE' if prediction == 1 else 'NON-HATE'

        features = exp.as_list(label=1)   # [(word, weight), ...]

        explanation = {
            'text': text,
            'prediction': label,
            'probability_non_hate': float(probs[0]),
            'probability_hate': float(probs[1]),
            'features': features,        # positive weight → pushes toward Hate
        }

        self._print_explanation(explanation)
        if plot:
            self._plot_explanation(explanation)

        return explanation

    def _print_explanation(self, exp: Dict):
        print("\n" + "=" * 80)
        print("LIME EXPLANATION - BASELINE MODEL (TF-IDF + Logistic Regression)")
        print("=" * 80)
        print(f'\nText: "{exp["text"]}"')
        print(f'\nPrediction: {exp["prediction"]}')
        print(f'  ├─ Probability (Non-Hate): {exp["probability_non_hate"]:.4f}')
        print(f'  └─ Probability (Hate):     {exp["probability_hate"]:.4f}\n')
        print(f'{"Rank":<6} {"Word":<22} {"Weight":<14} {"Impact"}')
        print("-" * 58)
        for i, (word, weight) in enumerate(exp['features'], 1):
            impact = "↑ Hate" if weight > 0 else "↓ Non-Hate"
            print(f'{i:<6} {word:<22} {weight:<14.6f} {impact}')

    def _plot_explanation(self, exp: Dict):
        features = exp['features']
        words = [f[0] for f in features]
        weights = [f[1] for f in features]

        colors = ['#d62728' if w > 0 else '#1f77b4' for w in weights]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(words, weights, color=colors, alpha=0.75)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('LIME Weight (positive → Hate, negative → Non-Hate)',
                      fontsize=11, fontweight='bold')
        ax.set_ylabel('Words', fontsize=11, fontweight='bold')
        snippet = exp['text'][:50] + ('...' if len(exp['text']) > 50 else '')
        ax.set_title(
            f'LIME Explanation: "{snippet}"\n'
            f'Prediction: {exp["prediction"]} '
            f'(Hate Prob: {exp["probability_hate"]:.4f})',
            fontsize=12, fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#d62728', alpha=0.75, label='Increases Hate Prediction'),
            Patch(facecolor='#1f77b4', alpha=0.75, label='Decreases Hate Prediction'),
        ], loc='best', fontsize=9)

        plt.tight_layout()
        plt.savefig('lime_explanation_baseline.png', dpi=300, bbox_inches='tight')
        print('\n✓ LIME explanation plot saved to: lime_explanation_baseline.png')
        plt.close()


class TransformerLimeExplainer:
    """
    LIME explainer for transformer models (XLM-RoBERTa, BERT, etc.).
    """

    def __init__(self, model: Any, tokenizer: Any,
                 class_names: List[str] = None, device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names or ['Non-Hate', 'Hate']

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()
        self._explainer = None

    def _predict_fn(self, texts: List[str]) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids'].to(self.device)
                attention_mask = enc['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                p = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                probs.append(p)
        return np.array(probs)

    def _get_explainer(self):
        if self._explainer is None:
            from lime.lime_text import LimeTextExplainer
            self._explainer = LimeTextExplainer(class_names=self.class_names)
        return self._explainer

    def explain_prediction(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 100,
        plot: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using LIME.

        Args:
            text: Input text to explain.
            num_features: Number of top words to highlight.
            num_samples: Perturbed samples (lower = faster for transformers).
            plot: Whether to save a bar-chart PNG.

        Returns:
            dict with keys: text, prediction, probability_hate,
                            probability_non_hate, features, important_tokens
        """
        explainer = self._get_explainer()
        exp = explainer.explain_instance(
            text,
            self._predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)
        )

        probs = self._predict_fn([text])[0]
        prediction = int(np.argmax(probs))
        label = 'HATE' if prediction == 1 else 'NON-HATE'

        features = exp.as_list(label=1)

        # Build important_tokens in the same shape as the SHAP explainer output
        # so the app can consume both transparently
        important_tokens = [
            {'token': word, 'importance': abs(weight), 'lime_weight': weight}
            for word, weight in features
        ]

        explanation = {
            'text': text,
            'prediction': label,
            'probability_non_hate': float(probs[0]),
            'probability_hate': float(probs[1]),
            'features': features,
            'important_tokens': important_tokens,
        }

        self._print_explanation(explanation)
        if plot:
            self._plot_explanation(explanation)

        return explanation

    def _print_explanation(self, exp: Dict):
        print("\n" + "=" * 80)
        print("LIME EXPLANATION - TRANSFORMER MODEL (XLM-RoBERTa)")
        print("=" * 80)
        print(f'\nText: "{exp["text"]}"')
        print(f'\nPrediction: {exp["prediction"]}')
        print(f'  ├─ Probability (Non-Hate): {exp["probability_non_hate"]:.4f}')
        print(f'  └─ Probability (Hate):     {exp["probability_hate"]:.4f}\n')
        print(f'{"Rank":<6} {"Word":<22} {"Weight":<14} {"Impact"}')
        print("-" * 58)
        for i, (word, weight) in enumerate(exp['features'], 1):
            impact = "↑ Hate" if weight > 0 else "↓ Non-Hate"
            print(f'{i:<6} {word:<22} {weight:<14.6f} {impact}')

    def _plot_explanation(self, exp: Dict):
        features = exp['features']
        words = [f[0] for f in features]
        weights = [f[1] for f in features]

        colors = ['#d62728' if w > 0 else '#1f77b4' for w in weights]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(words, weights, color=colors, alpha=0.75)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('LIME Weight (positive → Hate, negative → Non-Hate)',
                      fontsize=11, fontweight='bold')
        ax.set_ylabel('Words', fontsize=11, fontweight='bold')
        snippet = exp['text'][:50] + ('...' if len(exp['text']) > 50 else '')
        ax.set_title(
            f'LIME Explanation: "{snippet}"\n'
            f'Prediction: {exp["prediction"]} '
            f'(Hate Prob: {exp["probability_hate"]:.4f})',
            fontsize=12, fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#d62728', alpha=0.75, label='Increases Hate Prediction'),
            Patch(facecolor='#1f77b4', alpha=0.75, label='Decreases Hate Prediction'),
        ], loc='best', fontsize=9)

        plt.tight_layout()
        plt.savefig('lime_explanation_transformer.png', dpi=300, bbox_inches='tight')
        print('\n✓ LIME explanation plot saved to: lime_explanation_transformer.png')
        plt.close()


def explain_with_lime(
    text: str,
    model: Any,
    vectorizer_or_tokenizer: Any,
    model_type: str = 'auto',
    num_features: int = 10,
    num_samples: int = 500,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Unified interface for LIME-based explanation.

    Args:
        text: Text to explain.
        model: Trained model.
        vectorizer_or_tokenizer: TfidfVectorizer or HuggingFace tokenizer.
        model_type: 'baseline', 'transformer', or 'auto'.
        num_features: Number of top words to return.
        num_samples: LIME perturbation samples.
        plot: Save explanation plot.

    Returns:
        Explanation dict.
    """
    if model_type == 'auto':
        model_type = 'baseline' if hasattr(model, 'predict_proba') else 'transformer'

    if model_type == 'baseline':
        explainer = BaselineLimeExplainer(model, vectorizer_or_tokenizer)
        return explainer.explain_prediction(
            text, num_features=num_features, num_samples=num_samples, plot=plot
        )
    else:
        explainer = TransformerLimeExplainer(model, vectorizer_or_tokenizer)
        return explainer.explain_prediction(
            text, num_features=num_features, num_samples=num_samples, plot=plot
        )