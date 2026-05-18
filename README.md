# Hate Speech Detection System

> **BTech Major Project** — A severity-aware, explainable, context-sensitive, and self-improving hate speech detection system built on XLM-RoBERTa with an active-learning feedback loop.

---

## Research Contributions

| # | Contribution | Research Gap Addressed |
|---|---|---|
| 1 | **XLM-RoBERTa backbone** | English-centric models fail on Hinglish (Hindi+English code-mixed) social media |
| 2 | **Severity Score (0–100)** | Binary classification underrepresents the spectrum of harmful content |
| 3 | **Dual Explainability (SHAP + LIME)** | Black-box models give no rationale to content moderators |
| 4 | **Context-Aware Classification** | Single-sentence classifiers miss hate speech that requires thread context |
| 5 | **Active-Learning Feedback Loop** | Static models degrade over time as language evolves; no human-in-the-loop mechanism |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit Web App                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ Detector │  │   Batch   │  │ Feedback         │  │
│  │  Tab     │  │ Analysis  │  │ Dashboard        │  │
│  └────┬─────┘  └─────┬─────┘  └────────┬─────────┘  │
└───────┼──────────────┼────────────────┼─────────────┘
        │              │                │
        ▼              ▼                ▼
┌──────────────┐  ┌──────────┐  ┌──────────────────┐
│  Classifier  │  │  Batch   │  │  SQLite DB        │
│  Baseline or │  │ Predict  │  │  data/feedback.db │
│  XLM-RoBERTa │  └──────────┘  └────────┬─────────┘
└──────┬───────┘                          │
       │                         retrain_from_feedback.py
       ▼                                  │
┌──────────────────────────┐             ▼
│  Explainability          │   ┌─────────────────────┐
│  SHAP  |  LIME           │   │  Fine-tune loop     │
│  Attention Visualization │   │  (active learning)  │
└──────────────────────────┘   └─────────────────────┘
```

---

## Project Structure

```
hate-speech-detection/
├── app/
│   └── app.py                      # Streamlit UI (4 tabs)
├── data/
│   ├── raw/                        # Original datasets
│   └── feedback.db                 # SQLite corrections (auto-created)
├── models/                         # Saved model weights (auto-created by train.py)
│   ├── baseline_model.pkl
│   ├── baseline_vectorizer.pkl
│   ├── transformer_model/          # Fine-tuned XLM-RoBERTa
│   └── context_model/              # Context-aware XLM-RoBERTa
├── src/
│   ├── models/
│   │   ├── baseline.py             # TF-IDF + Logistic Regression
│   │   ├── transformer.py          # XLM-RoBERTa fine-tuning
│   │   └── context_model.py        # Conversation-context XLM-RoBERTa
│   ├── preprocessing/
│   │   ├── clean_text.py           # Text normalisation pipeline
│   │   └── load_data.py            # Dataset loader
│   ├── evaluation/
│   │   └── metrics.py              # F1, AUC-ROC, per-class metrics
│   └── explainability/
│       ├── shap_explainer.py       # SHAP word-level attribution
│       └── lime_explainer.py       # LIME local attribution
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis
├── train.py                        # Unified training CLI
├── main.py                         # Evaluation / comparison script
├── retrain_from_feedback.py        # Active-learning retraining script
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models

```bash
# Baseline (TF-IDF + Logistic Regression) — fast, ~30 seconds
python train.py --model baseline

# Transformer (XLM-RoBERTa) — GPU recommended
python train.py --model transformer --epochs 3 --batch_size 16

# Context-aware model (XLM-RoBERTa with conversation window)
python train.py --model context --epochs 3
```

### 3. Run the web app

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Retrain from moderator corrections

```bash
python retrain_from_feedback.py
```

---

## Model Performance

| Model | Accuracy | Macro F1 | Hate F1 | Notes |
|---|---|---|---|---|
| TF-IDF + LR (Baseline) | ~82% | ~0.78 | ~0.74 | English only |
| XLM-RoBERTa (fine-tuned) | ~89% | ~0.86 | ~0.84 | 100 languages |
| XLM-RoBERTa + Context | ~91% | ~0.88 | ~0.87 | Thread-aware |

> Results on `combined_hate_speech_dataset.csv`. Exact numbers in `metrics.json` and `evaluation_report.txt`.

---

## Research Gap Details

### Gap 1 — Multilingual / Hinglish Support
Standard BERT-base models are trained on English-dominated corpora.
XLM-RoBERTa (trained on 100 languages including Devanagari Hindi) significantly
outperforms English-only models on code-mixed Hinglish social media text.

**Key paper:** Conneau et al., 2020 — *Unsupervised Cross-lingual Representation Learning at Scale*

### Gap 2 — Severity Score Instead of Binary Label
Binary (hate / not-hate) classification collapses the continuum of harmful content.
This system outputs a **0–100 severity score** (P(hate) × 100), enabling
threshold-based routing: auto-remove >80, human-review 50–79, log 20–49.

**Key paper:** Vidgen & Derczynski, 2021 — *Directions in Abusive Language Training Data*

### Gap 3 — Dual Explainability (SHAP + LIME)
Most deployed systems are black boxes.  This project provides:
- **SHAP** — globally consistent, game-theory attribution
- **LIME** — fast local perturbation-based attribution
- Side-by-side comparison in the UI so moderators can spot ambiguous tokens

**Key papers:** Lundberg & Lee, 2017 (SHAP); Ribeiro et al., 2016 (LIME)

### Gap 4 — Context-Aware Classification
A reply like *"yes, exactly those people"* is not hateful alone but is in context.
The context model prepends up to N previous utterances using a `[CTX]` separator
token, allowing XLM-RoBERTa to attend across the conversation window.

**Key paper:** Pavlopoulos et al., 2020 — *Toxicity Detection: Does Context Really Matter?*

### Gap 5 — Active Learning Feedback Loop
Language evolves (new slang, dog-whistles). Static models degrade over time.
Every moderator correction in the Streamlit app is stored in `data/feedback.db`.
Running `retrain_from_feedback.py` merges corrections with the original training
set and fine-tunes the model, implementing a closed **human-in-the-loop** cycle.

**Key paper:** Settles, 2012 — *Active Learning (Synthesis Lectures on AI and ML)*

---

## Datasets

| Dataset | Source | Size | Labels |
|---|---|---|---|
| `trimmed_hate_speech_dataset.csv` | Twitter / public | ~25 K | hate / not-hate |
| `combined_hate_speech_dataset.csv` | Combined sources | ~50 K | hate / not-hate |
| `feedback.db` | Moderator UI | grows over time | corrected labels |

---

## Target Conference / Journal

- **ICTAI 2025** (IEEE International Conference on Tools with Artificial Intelligence)
- **ICON 2025** (International Conference on NLP)
- **ACL Findings** (workshop track on abusive language)

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{hatespeech2026,
  title  = {Severity-Aware Explainable Hate Speech Detection with Active Learning},
  author = {Verma, Chitraksh and Manasvee},
  year   = {2026},
  note   = {BTech Major Project, GitHub: Manasvee16/hate-speech-detection}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.