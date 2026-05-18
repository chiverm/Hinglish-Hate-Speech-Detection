"""
Hate Speech Detection — Streamlit App
Research contributions surfaced in the UI:
  1. Severity score (0-100) instead of binary label
  2. SHAP + LIME dual explainability with side-by-side comparison
  3. Context-aware classification (conversational window)
  4. Active-learning feedback loop: corrections stored in SQLite
     and surfaced on a Feedback Dashboard tab
"""

import sqlite3
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# ── project root on path ───────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.clean_text import clean_text
from src.models.baseline import predict_text as predict_baseline
from src.models.context_model import predict_with_context
from src.explainability.shap_explainer import explain_prediction
from src.explainability.lime_explainer import BaselineLimeExplainer, TransformerLimeExplainer

# ── SQLite feedback DB ─────────────────────────────────────────────────────────
_DB_PATH = project_root / 'data' / 'feedback.db'
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _init_db():
    con = sqlite3.connect(str(_DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            input_text  TEXT,
            model_type  TEXT,
            prediction  TEXT,
            severity    REAL,
            user_label  TEXT,
            comment     TEXT
        )
    """)
    con.commit()
    con.close()


def _save_feedback(input_text, model_type, prediction, severity, user_label, comment=""):
    con = sqlite3.connect(str(_DB_PATH))
    con.execute(
        "INSERT INTO feedback(ts,input_text,model_type,prediction,severity,user_label,comment) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            datetime.datetime.utcnow().isoformat(timespec='seconds'),
            input_text, model_type, prediction, severity, user_label, comment,
        )
    )
    con.commit()
    con.close()


def _load_feedback() -> pd.DataFrame:
    try:
        con = sqlite3.connect(str(_DB_PATH))
        df = pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", con)
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


_init_db()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { padding: 2rem; }
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.05rem;
}
.prediction-box {
    padding: 1.4rem; border-radius: .5rem; margin: .8rem 0; font-size: 1.1rem;
}
.hate-box    { background:#ffcccc; border-left:5px solid #ff0000; }
.non-hate-box{ background:#ccffcc; border-left:5px solid #00cc00; }
.severity-bar-outer {
    background:#e0e0e0; border-radius:8px; height:18px; width:100%;
}
.severity-bar-inner {
    height:18px; border-radius:8px;
    background: linear-gradient(90deg,#00cc00 0%,#ffcc00 50%,#ff0000 100%);
}
</style>
""", unsafe_allow_html=True)


# ── model loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_baseline_model():
    try:
        from src.models.baseline import train_baseline
        import pickle

        model_path      = project_root / 'models' / 'baseline_model.pkl'
        vectorizer_path = project_root / 'models' / 'baseline_vectorizer.pkl'

        if model_path.exists() and vectorizer_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            st.success("✓ Loaded saved baseline model")
        else:
            st.warning("⚠ Saved model not found — using demo model.")
            X_demo = ["I love this movie","You are so stupid","Great day",
                      "I hate you all","Amazing work","Go die","Beautiful",
                      "Seriously hate this","Wonderful","You should die"]
            y_demo = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            model, vectorizer = train_baseline(np.array(X_demo), np.array(y_demo))

        return model, vectorizer
    except Exception as exc:
        st.error(f"Error loading baseline model: {exc}")
        return None, None


@st.cache_resource
def load_transformer_model():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_path = project_root / 'models' / 'transformer_model'
        if model_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            st.success("✓ Loaded fine-tuned XLM-RoBERTa")
        else:
            st.warning("⚠ Fine-tuned model not found — loading XLM-RoBERTa base weights.")
            tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            model     = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base')

        return model, tokenizer
    except Exception as exc:
        st.error(f"Error loading transformer model: {exc}")
        return None, None


# ── helpers ───────────────────────────────────────────────────────────────────

def preprocess(text: str, keep_emojis: bool = False) -> str:
    return clean_text(text, keep_emojis=keep_emojis)


def severity_score(p_hate: float) -> int:
    """Map P(hate) → 0-100 severity score."""
    return min(100, int(round(p_hate * 100)))


def render_severity_bar(score: int):
    """Render an HTML gradient severity meter."""
    st.markdown(f"""
    <p style="margin-bottom:4px"><strong>Severity Score: {score}/100</strong></p>
    <div class="severity-bar-outer">
      <div class="severity-bar-inner" style="width:{score}%"></div>
    </div>
    <p style="font-size:.8rem;color:gray;margin-top:4px">
      0 = definitely safe &nbsp;|&nbsp; 100 = extreme hate speech
    </p>
    """, unsafe_allow_html=True)


def render_prediction_box(label: str, confidence: float):
    if label == "HATE":
        st.markdown(f"""
        <div class="prediction-box hate-box">
          <strong>🚨 HATE SPEECH DETECTED</strong><br>
          Confidence: <strong>{confidence:.2%}</strong>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box non-hate-box">
          <strong>✓ NON-HATE (Safe)</strong><br>
          Confidence: <strong>{confidence:.2%}</strong>
        </div>""", unsafe_allow_html=True)


def render_feature_table(features, method_label: str):
    """Render a ranked word-importance table."""
    rows = []
    for rank, (word, weight) in enumerate(features[:10], 1):
        if weight > 0:
            direction, color = "🔴 Hate",    "#d62728"
        else:
            direction, color = "🟢 Non-Hate", "#1f77b4"
        bar = "█" * min(int(abs(weight) * 80), 20)
        rows.append(
            f"| {rank} | **{word}** | "
            f"<span style='color:{color}'>{bar}</span> | "
            f"{direction} | `{weight:+.4f}` |"
        )
    header = (
        f"#### {method_label} — Top Contributing Words\n\n"
        "| # | Word | Weight Bar | Direction | Raw Weight |\n"
        "|---|------|-----------|-----------|------------|\n"
    )
    st.markdown(header + "\n".join(rows), unsafe_allow_html=True)


# ── main app ──────────────────────────────────────────────────────────────────

def main():
    st.title("🚨 Hate Speech Detection System")
    st.caption(
        "Severity-aware · Explainable (SHAP + LIME) · "
        "Context-aware · Active-learning feedback loop"
    )

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        model_type = st.radio(
            "Model:",
            ["Baseline (TF-IDF + LR)", "Transformer (XLM-RoBERTa)"],
            help="Baseline = fast; Transformer = accurate + multilingual",
        )
        do_preprocess = st.checkbox("Preprocess Text", value=True)
        keep_emojis   = st.checkbox("Keep Emojis",     value=False)
        use_context   = st.checkbox(
            "Context Mode",
            value=False,
            help="Prepend previous messages for conversational context",
        )
        show_expl = st.checkbox("Show Explanation", value=True)
        if show_expl:
            expl_method = st.selectbox(
                "Explanation Method:",
                ["SHAP", "LIME", "Both (SHAP + LIME)"],
            )
        else:
            expl_method = "SHAP"

        st.divider()
        st.markdown("""
**Research Contributions**
- 🌐 XLM-RoBERTa (Hinglish)
- 🎯 Severity score (0-100)
- 🔬 Dual explainability (SHAP + LIME)
- 💬 Conversation-context window
- 🔄 Active-learning feedback loop
""")

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Detector", "📚 Batch Analysis", "📊 Feedback Dashboard", "ℹ️ Help"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DETECTOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("Text Analysis")

        col_in, col_stat = st.columns([2, 1])
        with col_in:
            user_text = st.text_area(
                "Enter text to analyze:",
                placeholder="Type or paste your text here…\n"
                            "Supports English, Hindi, or Hinglish (code-mixed).",
                height=130,
                key="main_input",
            )
        with col_stat:
            st.markdown("**Input Stats**")
            if user_text:
                st.info(f"""
- **Length**: {len(user_text)} chars
- **Words**: {len(user_text.split())}
- **Status**: Ready ✓
""")
            else:
                st.warning("No text provided")

        # context inputs
        previous_messages: list = []
        if use_context:
            st.divider()
            st.subheader("📝 Conversation Context")
            n_ctx = st.number_input(
                "Previous messages to include (1–5):",
                min_value=1, max_value=5, value=2,
            )
            for i in range(n_ctx):
                msg = st.text_input(f"Message {i+1} (oldest first):", key=f"ctx_{i}")
                if msg:
                    previous_messages.append(msg)

        st.divider()
        analyse = st.button("🔍 Analyse Text", use_container_width=True, type="primary")

        if analyse:
            if not user_text or not user_text.strip():
                st.error("Please enter some text to analyse.")
            else:
                text_in = preprocess(user_text, keep_emojis) if do_preprocess else user_text
                if do_preprocess:
                    with st.expander("📝 Preprocessed text"):
                        st.write(text_in)

                with st.spinner(f"Running {model_type}…"):
                    if "Baseline" in model_type:
                        model, vectorizer = load_baseline_model()
                        if model is None:
                            st.stop()
                        result = predict_baseline(text_in, model, vectorizer)
                    else:
                        model, tokenizer = load_transformer_model()
                        if model is None:
                            st.stop()
                        if use_context:
                            result = predict_with_context(
                                text_in, model, tokenizer,
                                context=previous_messages,
                            )
                        else:
                            from src.models.transformer import predict_text
                            result = predict_text(text_in, model, tokenizer)

                label     = result['label']
                p_hate    = result['probability_hate']
                p_nonhate = result['probability_non_hate']
                confidence = p_hate if label == 'HATE' else p_nonhate
                sev        = severity_score(p_hate)

                # ── results ──────────────────────────────────────────────────
                st.subheader("📊 Analysis Results")
                render_prediction_box(label, confidence)

                c1, c2, c3 = st.columns(3)
                c1.metric("Hate Probability",    f"{p_hate:.2%}")
                c2.metric("Non-Hate Probability", f"{p_nonhate:.2%}")
                c3.metric("Severity Score",       f"{sev}/100")

                render_severity_bar(sev)

                # ── explainability ───────────────────────────────────────────
                if show_expl:
                    st.divider()

                    def _run_shap():
                        st.subheader("🔬 SHAP Explanation")
                        st.caption(
                            "SHapley Additive exPlanations — game-theory attribution "
                            "that is globally consistent across the dataset."
                        )
                        try:
                            with st.spinner("Computing SHAP values…"):
                                if "Baseline" in model_type:
                                    exp      = explain_prediction(
                                        text_in, model, vectorizer,
                                        model_type='baseline', plot=False,
                                    )
                                    features = [
                                        (f['feature'], f['shap_value'])
                                        for f in exp.get('important_features', [])
                                    ]
                                else:
                                    exp      = explain_prediction(
                                        text_in, model, tokenizer,
                                        model_type='transformer', plot=False,
                                    )
                                    features = [
                                        (t['token'], t['importance'])
                                        for t in exp.get('important_tokens', [])
                                    ]
                            render_feature_table(features, "SHAP")
                        except Exception as exc:
                            st.warning(f"SHAP error: {exc}")

                    def _run_lime():
                        st.subheader("🟡 LIME Explanation")
                        st.caption(
                            "Local Interpretable Model-agnostic Explanations — "
                            "perturbs the input locally to identify influential words."
                        )
                        try:
                            with st.spinner("Computing LIME values…"):
                                if "Baseline" in model_type:
                                    lime_exp    = BaselineLimeExplainer(model, vectorizer)
                                    lime_result = lime_exp.explain_prediction(
                                        text_in, num_features=10,
                                        num_samples=300, plot=False,
                                    )
                                else:
                                    lime_exp    = TransformerLimeExplainer(model, tokenizer)
                                    lime_result = lime_exp.explain_prediction(
                                        text_in, num_features=10,
                                        num_samples=80, plot=False,
                                    )
                            render_feature_table(lime_result['features'], "LIME")
                        except Exception as exc:
                            st.warning(f"LIME error: {exc}")

                    if expl_method == "SHAP":
                        _run_shap()
                    elif expl_method == "LIME":
                        _run_lime()
                    else:
                        col_s, col_l = st.columns(2)
                        with col_s:
                            _run_shap()
                        with col_l:
                            _run_lime()
                        st.info(
                            "💡 **Research insight:** When SHAP and LIME agree on the "
                            "top tokens, the prediction is more trustworthy. "
                            "Disagreements reveal context-sensitive or ambiguous words."
                        )

                # ── active-learning feedback ──────────────────────────────────
                st.divider()
                st.subheader("🔄 Feedback (Active Learning)")
                st.caption(
                    "If this prediction is **wrong**, correct it here. "
                    "Corrections are stored and used to retrain the model."
                )
                with st.form("feedback_form", clear_on_submit=True):
                    correct_label = st.radio(
                        "Correct label:",
                        ["Model is correct ✓", "HATE", "NON-HATE"],
                        horizontal=True,
                    )
                    comment = st.text_input(
                        "Optional comment (e.g. 'sarcasm', 'Hinglish idiom'):",
                    )
                    submitted = st.form_submit_button("Submit Feedback")

                if submitted:
                    user_label = (
                        label
                        if correct_label == "Model is correct ✓"
                        else correct_label
                    )
                    _save_feedback(
                        input_text=text_in,
                        model_type=model_type,
                        prediction=label,
                        severity=sev,
                        user_label=user_label,
                        comment=comment,
                    )
                    if correct_label == "Model is correct ✓":
                        st.success("✓ Positive feedback saved — thank you!")
                    else:
                        st.warning(
                            f"✓ Correction saved: model said **{label}**, "
                            f"you said **{user_label}**. "
                            "This will be used in the next retraining cycle."
                        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — BATCH ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("📚 Batch Analysis")
        batch_input = st.text_area(
            "Enter texts (one per line):",
            placeholder="Text 1\nText 2\nText 3",
            height=200,
        )

        if st.button("Analyse Batch", use_container_width=True):
            if not batch_input.strip():
                st.error("Please enter at least one line.")
            else:
                texts = [t.strip() for t in batch_input.splitlines() if t.strip()]

                with st.spinner(f"Analysing {len(texts)} texts…"):
                    if "Baseline" in model_type:
                        model, vectorizer = load_baseline_model()
                        rows = []
                        for t in texts:
                            r = predict_baseline(
                                preprocess(t) if do_preprocess else t,
                                model, vectorizer,
                            )
                            rows.append({
                                'Text':       (t[:60] + "…") if len(t) > 60 else t,
                                'Prediction': r['label'],
                                'Severity':   severity_score(r['probability_hate']),
                                'P(Hate)':    f"{r['probability_hate']:.2%}",
                            })
                    else:
                        from src.models.transformer import predict_batch
                        model, tokenizer = load_transformer_model()
                        cleaned = [preprocess(t) if do_preprocess else t for t in texts]
                        probs, preds = predict_batch(
                            model, tokenizer, cleaned,
                            batch_size=16, max_length=128,
                        )
                        rows = [
                            {
                                'Text':       (t[:60] + "…") if len(t) > 60 else t,
                                'Prediction': 'HATE' if p == 1 else 'NON-HATE',
                                'Severity':   severity_score(pr),
                                'P(Hate)':    f"{pr:.2%}",
                            }
                            for t, p, pr in zip(texts, preds, probs)
                        ]

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                hate_n = sum(1 for r in rows if r['Prediction'] == 'HATE')
                avg_sev = np.mean([r['Severity'] for r in rows])
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Texts",    len(rows))
                c2.metric("Hate Speech",    f"{hate_n} ({hate_n/len(rows):.0%})")
                c3.metric("Avg Severity",   f"{avg_sev:.0f}/100")

                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇ Download CSV",
                    data=csv,
                    file_name="batch_results.csv",
                    mime="text/csv",
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — FEEDBACK DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("📊 Feedback Dashboard")
        st.caption(
            "Live view of all moderator corrections — drives the active-learning pipeline. "
            "A retraining script reads `data/feedback.db` periodically."
        )

        df_fb = _load_feedback()

        if df_fb.empty:
            st.info("No feedback collected yet. Use the Detector tab to analyse text and submit corrections.")
        else:
            total      = len(df_fb)
            errors     = df_fb[df_fb['prediction'] != df_fb['user_label']]
            n_errors   = len(errors)
            error_rate = n_errors / total if total else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Submissions",  total)
            c2.metric("Corrections",        n_errors)
            c3.metric("Error Rate",         f"{error_rate:.1%}")
            c4.metric("Avg Severity",       f"{df_fb['severity'].mean():.0f}/100")

            st.divider()

            # error breakdown by model
            st.subheader("Corrections by Model Type")
            if n_errors:
                err_counts = errors.groupby('model_type').size().reset_index(name='Corrections')
                st.bar_chart(err_counts.set_index('model_type'))
            else:
                st.success("No corrections recorded — model predictions all confirmed correct so far!")

            st.subheader("Correction Direction")
            if n_errors:
                errors['direction'] = (
                    errors['prediction'] + " → " + errors['user_label']
                )
                st.bar_chart(errors['direction'].value_counts())

            st.subheader("Severity Distribution of Corrections")
            if n_errors:
                st.bar_chart(
                    errors['severity'].apply(
                        lambda s: f"{int(s)//10*10}-{int(s)//10*10+9}"
                    ).value_counts().sort_index()
                )

            st.divider()
            st.subheader("All Feedback Records")
            display_df = df_fb[['ts','model_type','prediction','user_label',
                                  'severity','comment','input_text']].copy()
            display_df.columns = ['Timestamp','Model','Prediction','User Label',
                                   'Severity','Comment','Input Text']
            st.dataframe(display_df, use_container_width=True)

            csv_fb = df_fb.to_csv(index=False)
            st.download_button(
                "⬇ Export Feedback CSV",
                data=csv_fb,
                file_name="feedback_export.csv",
                mime="text/csv",
            )

            st.divider()
            st.info(
                "**Active Learning Retraining** — run `python retrain_from_feedback.py` "
                "to fine-tune the model on the corrections above. "
                "The script is generated automatically alongside this app."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — HELP
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("ℹ️ Help & Research Background")
        st.markdown("""
### How It Works
1. **Input** — Enter text in English, Hindi, or code-mixed Hinglish
2. **Preprocessing** — Strips URLs, @mentions, #hashtags, normalises casing
3. **Detection** — ML model predicts hate vs. non-hate with a **severity score**
4. **Explanation** — SHAP and/or LIME highlight which words drove the decision
5. **Feedback** — If wrong, correct it; corrections are stored for retraining

---

### Model Architectures

| Model | Speed | Multilingual | Hinglish |
|---|---|---|---|
| TF-IDF + Logistic Regression | ⚡ Fast | ✗ | Limited |
| XLM-RoBERTa (fine-tuned) | 🐢 Slower | ✅ 100 langs | ✅ |

---

### Severity Score
Rather than a hard binary label, the **severity score (0–100)** maps P(hate)
to a continuous risk scale. This addresses the research gap that binary
classification underrepresents the spectrum of harmful content.

| Range | Interpretation |
|---|---|
| 0–20 | Safe / benign |
| 21–50 | Mildly offensive, review recommended |
| 51–79 | Likely hate speech |
| 80–100 | Extreme / targeted hate speech |

---

### Explainability Methods

**SHAP (SHapley Additive exPlanations)**
- Game-theory-based attribution
- Globally consistent across the full dataset
- 🔴 Red = pushes toward Hate | 🟢 Green = pushes toward Non-Hate

**LIME (Local Interpretable Model-agnostic Explanations)**
- Perturbs the input locally to build a linear approximation
- Faster for transformers (no gradient computation)
- Captures locally influential tokens even when SHAP misses them

**Research insight:** Showing both methods side-by-side lets moderators
spot *disagreements* — tokens that are context-sensitive or ambiguous.

---

### Context Mode
Social media hate speech is often only recognisable in thread context.
Enabling **Context Mode** prepends up to 5 previous messages using a
`[CTX]` separator, which the XLM-RoBERTa model has been trained to attend to.

*Paper reference: Pavlopoulos et al., 2020 — "Toxicity Detection: Does Context Really Matter?"*

---

### Active Learning Feedback Loop
Every moderator correction is stored in `data/feedback.db` (SQLite).
Running `retrain_from_feedback.py` merges these corrections back into
the training set and triggers a fine-tuning pass, implementing the
**human-in-the-loop active learning** pipeline described in the project report.

---

### Supported Languages
- English (EN)
- Hindi (HI) — Devanagari and romanised
- Code-mixed Hinglish (EN-HI)
""")
        st.divider()
        st.markdown("""
**Built with:** Streamlit · HuggingFace Transformers · XLM-RoBERTa ·
SHAP · LIME · scikit-learn · PyTorch · SQLite
""")


if __name__ == "__main__":
    main()