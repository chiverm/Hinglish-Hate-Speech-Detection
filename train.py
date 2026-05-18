"""
train.py — Research-grade training pipeline for hate speech detection.

Primary thesis: imbalance-aware multilingual transformer for noisy
Hinglish / code-mixed social media text, with explanation-backed
error analysis.

Usage examples:
    python train.py --model baseline
    python train.py --model baseline --tfidf-char
    python train.py --model transformer --loss ce  --cv-folds 5
    python train.py --model transformer --loss focal --cv-folds 5
    python train.py --model all --loss focal --cv-folds 5
    python train.py --model context --loss focal
    python train.py --model transformer --loss focal \\
        --lr 2e-5 --epochs 4 --batch-size 16 --max-length 128 \\
        --dropout 0.1 --weight-decay 0.01 --scheduler linear \\
        --early-stopping-patience 2 --threshold 0.45 --seed 42
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc as sklearn_auc,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean_text import clean_text, is_hinglish

RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR  = PROJECT_ROOT / 'models'
METRICS_OUT = PROJECT_ROOT / 'metrics.json'
REPORT_OUT  = PROJECT_ROOT / 'evaluation_report.txt'


def _make_dirs():
    for d in (RESULTS_DIR, MODELS_DIR,
              RESULTS_DIR / 'plots',
              RESULTS_DIR / 'errors'):
        d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _load_dataset(data_path: Path):
    """Load CSV, clean, deduplicate, tag Hinglish."""
    print(f"\n▶  Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = [c.lower().strip() for c in df.columns]

    text_col  = next((c for c in df.columns if 'text'  in c), None)
    label_col = next((c for c in df.columns if 'label' in c or 'class' in c), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Cannot find text/label columns in {data_path}. "
            f"Found: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].dropna().rename(
        columns={text_col: 'text_raw', label_col: 'label'}
    )
    df['label'] = df['label'].apply(lambda x: 0 if int(x) == 0 else 1)
    df['text']  = df['text_raw'].apply(clean_text)
    df = df[df['text'].str.strip() != ''].reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  ⚠  Removed {removed:,} near-duplicate rows.")

    df['is_hinglish'] = df['text_raw'].apply(is_hinglish)

    n_hate = int(df['label'].sum())
    n_non  = int((df['label'] == 0).sum())
    n_hin  = int(df['is_hinglish'].sum())
    print(f"  Samples  : {len(df):,}")
    print(f"  Hate     : {n_hate:,}  ({n_hate/len(df)*100:.1f}%)")
    print(f"  Non-hate : {n_non:,}  ({n_non/len(df)*100:.1f}%)")
    print(f"  Hinglish : {n_hin:,}  ({n_hin/len(df)*100:.1f}%)")
    return df


def _holdout_split(df: pd.DataFrame, test_size: float = 0.15, seed: int = 42):
    """Carve out a single untouched holdout before any CV."""
    df_tv, df_test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label']
    )
    print(f"\n  Train+Val: {len(df_tv):,}  |  Holdout Test: {len(df_test):,}")
    return df_tv.reset_index(drop=True), df_test.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_metrics(y_true, y_pred, y_prob=None, threshold: float = 0.5):
    if y_prob is not None:
        y_pred = (np.array(y_prob) >= threshold).astype(int)
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    hate_f1  = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    rep = classification_report(
        y_true, y_pred, target_names=['Non-Hate', 'Hate'],
        output_dict=True, zero_division=0,
    )
    result = {
        'accuracy'  : round(acc, 4),
        'macro_f1'  : round(macro_f1, 4),
        'hate_f1'   : round(hate_f1, 4),
        'hate_prec' : round(rep['Hate']['precision'], 4),
        'hate_rec'  : round(rep['Hate']['recall'],    4),
        'threshold' : threshold,
        'report'    : rep,
        'y_pred'    : list(y_pred),
    }
    if y_prob is not None:
        pa = np.array(y_prob)
        try:
            result['roc_auc'] = round(roc_auc_score(y_true, pa), 4)
        except Exception:
            result['roc_auc'] = None
        try:
            prec, rec, _ = precision_recall_curve(y_true, pa)
            result['pr_auc'] = round(sklearn_auc(rec, prec), 4)
        except Exception:
            result['pr_auc'] = None
    return result


def _print_metrics(m: dict, tag: str = ''):
    lbl = f" [{tag}]" if tag else ""
    print(f"  Accuracy{lbl}  : {m['accuracy']:.4f}")
    print(f"  Macro-F1{lbl}  : {m['macro_f1']:.4f}")
    print(f"  Hate-F1{lbl}   : {m['hate_f1']:.4f}  "
          f"(P={m['hate_prec']:.3f}  R={m['hate_rec']:.3f})")
    if m.get('roc_auc'):
        print(f"  ROC-AUC{lbl}   : {m['roc_auc']:.4f}")
    if m.get('pr_auc'):
        print(f"  PR-AUC{lbl}    : {m['pr_auc']:.4f}")


def _threshold_sweep(y_true, y_prob, tag: str = 'model'):
    """Sweep threshold 0.25→0.75, return best threshold by hate-F1."""
    print(f"\n  ── Threshold sweep [{tag}] ──")
    best_t, best_f1 = 0.5, 0.0
    rows = []
    for t in np.arange(0.25, 0.76, 0.05):
        preds = (np.array(y_prob) >= t).astype(int)
        hf1   = f1_score(y_true, preds, pos_label=1, average='binary', zero_division=0)
        mf1   = f1_score(y_true, preds, average='macro', zero_division=0)
        rows.append({'threshold': round(float(t), 2), 'hate_f1': round(hf1, 4),
                     'macro_f1': round(mf1, 4)})
        marker = '  ◀ best' if hf1 > best_f1 else ''
        print(f"    t={t:.2f}  hate-F1={hf1:.4f}  macro-F1={mf1:.4f}{marker}")
        if hf1 > best_f1:
            best_f1, best_t = hf1, float(t)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f'threshold_sweep_{tag}.csv', index=False)
    print(f"  Best threshold: {best_t:.2f}  (hate-F1 = {best_f1:.4f})")
    return best_t


def _plot_cm(y_true, y_pred, tag: str, split: str = 'test'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Non-Hate', 'Hate']).plot(
        ax=ax, colorbar=False)
    ax.set_title(f'{tag} — Confusion Matrix ({split})')
    out = RESULTS_DIR / 'plots' / f'cm_{tag}_{split}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Confusion matrix → {out}")


def _plot_pr_roc(y_true, y_prob, tag: str):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(rec, prec, lw=2)
        axes[0].set(xlabel='Recall', ylabel='Precision',
                    title=f'PR Curve (AUC={sklearn_auc(rec, prec):.4f})',
                    xlim=[0, 1], ylim=[0, 1])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[1].plot(fpr, tpr, lw=2)
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set(xlabel='FPR', ylabel='TPR',
                    title=f'ROC (AUC={roc_auc_score(y_true, y_prob):.4f})',
                    xlim=[0, 1], ylim=[0, 1])
        fig.suptitle(tag, fontsize=13)
        out = RESULTS_DIR / 'plots' / f'pr_roc_{tag}.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ PR/ROC curves → {out}")
    except Exception as e:
        print(f"  ⚠  Could not plot PR/ROC: {e}")


def _save_errors(texts_raw, y_true, y_pred, y_prob=None,
                 tag: str = 'model', n: int = 50):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    texts_raw = np.array(texts_raw)
    rows = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            rows.append({
                'text'      : texts_raw[i],
                'true_label': int(y_true[i]),
                'pred_label': int(y_pred[i]),
                'error_type': 'FP' if y_pred[i] == 1 else 'FN',
                'prob_hate' : round(float(y_prob[i]), 4) if y_prob is not None else None,
            })
    if not rows:
        return
    df_err = pd.DataFrame(rows)
    if y_prob is not None:
        df_err['_abs'] = (df_err['prob_hate'] - 0.5).abs()
        df_err = df_err.sort_values('_abs', ascending=False).drop(
            columns=['_abs']).head(n)
    out = RESULTS_DIR / 'errors' / f'errors_{tag}.csv'
    df_err.to_csv(out, index=False)
    print(f"  ✓ Error samples → {out}  "
          f"(FP={int((df_err['error_type']=='FP').sum())}, "
          f"FN={int((df_err['error_type']=='FN').sum())})")


# ═══════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════════

def focal_loss_fn(logits, labels, gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Downweights easy examples, focuses on hard/ambiguous ones.
    """
    import torch
    import torch.nn.functional as F
    ce     = F.cross_entropy(logits, labels, reduction='none')
    pt     = torch.exp(-ce)
    at     = torch.where(labels == 1,
                         torch.full_like(ce, 1 - alpha),
                         torch.full_like(ce, alpha))
    return (at * (1 - pt) ** gamma * ce).mean()


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def train_baseline_model(df_tv, df_test, args):
    from src.models.baseline import train_baseline, save_baseline, evaluate_baseline

    variant = 'word+char' if args.tfidf_char else 'word'
    tag     = f'baseline_{variant}'
    print("\n" + "=" * 64)
    print(f"  BASELINE  TF-IDF ({variant.upper()}) + Logistic Regression")
    print("=" * 64)

    X_all  = df_tv['text'].tolist()
    y_all  = df_tv['label'].tolist()
    X_test = df_test['text'].tolist()
    y_test = df_test['label'].tolist()

    cv_result = {}
    if args.cv_folds > 1:
        print(f"\n▶  {args.cv_folds}-fold stratified CV …")
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                              random_state=args.seed)
        fold_macro, fold_hate = [], []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), 1):
            X_tr = [X_all[i] for i in tr_idx];  y_tr = [y_all[i] for i in tr_idx]
            X_va = [X_all[i] for i in va_idx];  y_va = [y_all[i] for i in va_idx]
            m, v = train_baseline(X_tr, y_tr,
                                  max_features=50_000,
                                  ngram_range=(1, 3) if args.tfidf_char else (1, 2),
                                  use_char_ngrams=args.tfidf_char,
                                  C=1.0, class_weight='balanced')
            vm = evaluate_baseline(m, v, X_va, y_va)
            fold_macro.append(vm['macro_f1'])
            fold_hate.append(vm['hate_f1'])
            print(f"  Fold {fold}: macro-F1={vm['macro_f1']:.4f}  hate-F1={vm['hate_f1']:.4f}")
        print(f"\n  CV macro-F1 : {np.mean(fold_macro):.4f} ± {np.std(fold_macro):.4f}")
        print(f"  CV hate-F1  : {np.mean(fold_hate):.4f} ± {np.std(fold_hate):.4f}")
        cv_result = {
            'cv_macro_f1_mean': round(float(np.mean(fold_macro)), 4),
            'cv_macro_f1_std' : round(float(np.std(fold_macro)),  4),
            'cv_hate_f1_mean' : round(float(np.mean(fold_hate)),  4),
            'cv_hate_f1_std'  : round(float(np.std(fold_hate)),   4),
        }

    print("\n▶  Training on full train+val …")
    model, vectorizer = train_baseline(
        X_all, y_all,
        max_features=50_000,
        ngram_range=(1, 3) if args.tfidf_char else (1, 2),
        use_char_ngrams=args.tfidf_char,
        C=1.0, class_weight='balanced',
    )
    save_baseline(model, vectorizer, suffix=tag)

    print("\n▶  Holdout test …")
    tm = evaluate_baseline(model, vectorizer, X_test, y_test)
    _print_metrics(tm, tag)
    _plot_cm(y_test, tm['y_pred'], tag)

    # Hinglish subset
    df_hin = df_test[df_test['is_hinglish']]
    hin_result = {}
    if len(df_hin) >= 10:
        print(f"\n▶  Hinglish subset (n={len(df_hin)}) …")
        hm = evaluate_baseline(model, vectorizer,
                               df_hin['text'].tolist(),
                               df_hin['label'].tolist())
        _print_metrics(hm, f'{tag}_hinglish')
        hin_result = {'hinglish_macro_f1': hm['macro_f1'],
                      'hinglish_hate_f1' : hm['hate_f1']}

    _save_errors(df_test['text_raw'].tolist(), y_test, tm['y_pred'], tag=tag)

    return {**cv_result, **hin_result, 'test': _serializable(tm)}


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════

def train_transformer_model(df_tv, df_test, args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    )

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = f'transformer_{args.loss}'
    print("\n" + "=" * 64)
    print(f"  TRANSFORMER  {args.base_model}  loss={args.loss.upper()}  device={DEVICE}")
    print("=" * 64)

    class HSD(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.enc = tokenizer(texts, padding='max_length', truncation=True,
                                 max_length=max_len, return_tensors='pt')
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {'input_ids'     : self.enc['input_ids'][i],
                    'attention_mask': self.enc['attention_mask'][i],
                    'labels'        : self.labels[i]}

    def _loss_fn(logits, labels, cw):
        import torch.nn.functional as F
        if args.loss == 'focal':
            return focal_loss_fn(logits, labels,
                                 gamma=args.focal_gamma, alpha=args.focal_alpha)
        return F.cross_entropy(logits, labels, weight=cw)

    def _run(X_tr, y_tr, X_va, y_va, fold_tag=''):
        tok = AutoTokenizer.from_pretrained(args.base_model)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=2,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
        ).to(DEVICE)

        tr_dl = DataLoader(HSD(X_tr, y_tr, tok, args.max_length),
                           batch_size=args.batch_size, shuffle=True, num_workers=0)
        va_dl = DataLoader(HSD(X_va, y_va, tok, args.max_length),
                           batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

        counts = np.bincount(y_tr)
        cw = torch.tensor([1.0 / c for c in counts], dtype=torch.float).to(DEVICE)
        cw = cw / cw.sum() * len(counts)

        opt   = torch.optim.AdamW(mdl.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
        total = len(tr_dl) * args.epochs
        warm  = max(1, int(0.06 * total))
        sched = (get_cosine_schedule_with_warmup if args.scheduler == 'cosine'
                 else get_linear_schedule_with_warmup)(opt, warm, total)

        best_f1, patience, best_state = -1.0, 0, None

        for ep in range(1, args.epochs + 1):
            mdl.train()
            ep_loss = 0.0
            for batch in tr_dl:
                opt.zero_grad()
                ids  = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                lbl  = batch['labels'].to(DEVICE)
                out  = mdl(input_ids=ids, attention_mask=mask)
                loss = _loss_fn(out.logits, lbl, cw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step()
                ep_loss += loss.item()

            mdl.eval()
            vp, vprob = [], []
            with torch.no_grad():
                for batch in va_dl:
                    out  = mdl(input_ids=batch['input_ids'].to(DEVICE),
                               attention_mask=batch['attention_mask'].to(DEVICE))
                    vprob.extend(torch.softmax(out.logits, -1)[:, 1].cpu().tolist())
                    vp.extend(out.logits.argmax(-1).cpu().tolist())

            vf1 = f1_score(y_va, vp, average='macro', zero_division=0)
            print(f"  Ep {ep}/{args.epochs} {fold_tag}  "
                  f"loss={ep_loss/len(tr_dl):.4f}  val macro-F1={vf1:.4f}")

            if vf1 > best_f1:
                best_f1 = vf1; patience = 0
                best_state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
            else:
                patience += 1
                if patience >= args.early_stopping_patience:
                    print(f"  ↳ Early stop at epoch {ep}.")
                    break

        mdl.load_state_dict(best_state); mdl.eval()
        return mdl, tok, vprob, vp

    def _predict(texts, mdl, tok):
        ds = HSD(texts, [0] * len(texts), tok, args.max_length)
        dl = DataLoader(ds, batch_size=args.batch_size * 2,
                        shuffle=False, num_workers=0)
        mdl.eval(); probs, preds = [], []
        with torch.no_grad():
            for batch in dl:
                out = mdl(input_ids=batch['input_ids'].to(DEVICE),
                          attention_mask=batch['attention_mask'].to(DEVICE))
                probs.extend(torch.softmax(out.logits, -1)[:, 1].cpu().tolist())
                preds.extend(out.logits.argmax(-1).cpu().tolist())
        return probs, preds

    X_all  = df_tv['text'].tolist();   y_all  = df_tv['label'].tolist()
    X_test = df_test['text'].tolist(); y_test = df_test['label'].tolist()

    # ── cross-validation ──────────────────────────────────────────────────
    cv_result = {}
    if args.cv_folds > 1:
        print(f"\n▶  {args.cv_folds}-fold stratified CV …")
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                              random_state=args.seed)
        fold_macro, fold_hate = [], []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), 1):
            X_tr = [X_all[i] for i in tr_idx]; y_tr = [y_all[i] for i in tr_idx]
            X_va = [X_all[i] for i in va_idx]; y_va = [y_all[i] for i in va_idx]
            _, _, vprob, vpred = _run(X_tr, y_tr, X_va, y_va,
                                      fold_tag=f'[fold {fold}/{args.cv_folds}]')
            vm = _compute_metrics(y_va, vpred, vprob, threshold=args.threshold)
            fold_macro.append(vm['macro_f1'])
            fold_hate.append(vm['hate_f1'])
            print(f"  Fold {fold}: macro-F1={vm['macro_f1']:.4f}  hate-F1={vm['hate_f1']:.4f}")
        print(f"\n  CV macro-F1 : {np.mean(fold_macro):.4f} ± {np.std(fold_macro):.4f}")
        print(f"  CV hate-F1  : {np.mean(fold_hate):.4f} ± {np.std(fold_hate):.4f}")
        cv_result = {
            'cv_macro_f1_mean': round(float(np.mean(fold_macro)), 4),
            'cv_macro_f1_std' : round(float(np.std(fold_macro)),  4),
            'cv_hate_f1_mean' : round(float(np.mean(fold_hate)),  4),
            'cv_hate_f1_std'  : round(float(np.std(fold_hate)),   4),
        }

    # ── final train on full trainval ──────────────────────────────────────
    print("\n▶  Final training on full train+val …")
    X_tv_list = X_all
    y_tv_list = y_all
    # use a small validation slice for early stopping (no fold leakage)
    X_ftv, X_fva, y_ftv, y_fva = train_test_split(
        X_tv_list, y_tv_list, test_size=0.1,
        random_state=args.seed, stratify=y_tv_list
    )
    final_model, final_tok, val_probs, val_preds = _run(
        X_ftv, y_ftv, X_fva, y_fva, fold_tag='[final]'
    )

    # threshold sweep on validation slice
    best_t = _threshold_sweep(y_fva, val_probs, tag=tag)
    eff_threshold = args.threshold if args.threshold != 0.5 else best_t

    # save model
    mdl_out = MODELS_DIR / tag
    final_model.save_pretrained(str(mdl_out))
    final_tok.save_pretrained(str(mdl_out))
    print(f"  ✓ Model saved → {mdl_out}")

    # ── holdout test ──────────────────────────────────────────────────────
    print("\n▶  Holdout test …")
    test_probs, test_preds = _predict(X_test, final_model, final_tok)
    tm = _compute_metrics(y_test, test_preds, test_probs,
                          threshold=eff_threshold)
    _print_metrics(tm, tag)
    _plot_cm(y_test, tm['y_pred'], tag)
    _plot_pr_roc(y_test, test_probs, tag)

    # Hinglish subset
    df_hin = df_test[df_test['is_hinglish']]
    hin_result = {}
    if len(df_hin) >= 10:
        print(f"\n▶  Hinglish subset (n={len(df_hin)}) …")
        hp, hpred = _predict(df_hin['text'].tolist(), final_model, final_tok)
        hm = _compute_metrics(df_hin['label'].tolist(), hpred, hp,
                               threshold=eff_threshold)
        _print_metrics(hm, f'{tag}_hinglish')
        hin_result = {'hinglish_macro_f1': hm['macro_f1'],
                      'hinglish_hate_f1' : hm['hate_f1']}

    _save_errors(df_test['text_raw'].tolist(), y_test,
                 tm['y_pred'], test_probs, tag=tag)

    return {**cv_result, **hin_result,
            'test': _serializable(tm), 'best_threshold': eff_threshold}


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT MODEL  (exploratory, not headline contribution)
# ═══════════════════════════════════════════════════════════════════════════

def train_context_model(df_tv, df_test, args):
    """
    Exploratory context augmentation: each sample is prepended with the
    previous 2 samples as pseudo-context.
    NOTE: this uses synthetic context (adjacent dataset rows), NOT real
    conversation threads. Frame as an exploratory experiment only.
    """
    from src.models.context_model import build_context_text, fine_tune_context
    from src.models.transformer import predict_batch
    from sklearn.metrics import accuracy_score, classification_report

    tag = f'context_{args.loss}'
    print("\n" + "=" * 64)
    print("  CONTEXT MODEL (exploratory) — pseudo-context window=2")
    print("=" * 64)

    def _add_ctx(texts, window=2):
        return [build_context_text(t, texts[max(0, i-window):i], window)
                for i, t in enumerate(texts)]

    X_all  = df_tv['text'].tolist();   y_all  = df_tv['label'].tolist()
    X_test = df_test['text'].tolist(); y_test = df_test['label'].tolist()

    X_tv_ctx   = _add_ctx(X_all)
    X_test_ctx = _add_ctx(X_test)

    X_ftv, X_fva, y_ftv, y_fva = train_test_split(
        X_tv_ctx, y_all, test_size=0.1,
        random_state=args.seed, stratify=y_all
    )

    try:
        model, tok = fine_tune_context(
            X_ftv, y_ftv, X_fva, y_fva,
            base_model=args.base_model,
            loss=args.loss,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
        )
        test_probs, test_preds = predict_batch(
            model, tok, X_test_ctx,
            batch_size=args.batch_size * 2,
            max_length=args.max_length,
        )
        tm = _compute_metrics(y_test, test_preds, test_probs,
                              threshold=args.threshold)
        _print_metrics(tm, tag)
        _plot_cm(y_test, tm['y_pred'], tag)
        _save_errors(df_test['text_raw'].tolist(), y_test,
                     tm['y_pred'], test_probs, tag=tag)
        return {'test': _serializable(tm)}
    except Exception as exc:
        print(f"  ⚠  Context model failed: {exc}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# ABLATION TABLE
# ═══════════════════════════════════════════════════════════════════════════

def _print_ablation_table(all_results: dict):
    """Print a compact ablation table to stdout and save as CSV."""
    rows = []
    for model_tag, res in all_results.items():
        test = res.get('test', {})
        rows.append({
            'Model'           : model_tag,
            'CV macro-F1'     : f"{res.get('cv_macro_f1_mean', '-')}"
                                f" ± {res.get('cv_macro_f1_std', '-')}",
            'Test macro-F1'   : test.get('macro_f1', '-'),
            'Test hate-F1'    : test.get('hate_f1',  '-'),
            'Test Prec (hate)': test.get('hate_prec','-'),
            'Test Rec (hate)' : test.get('hate_rec', '-'),
            'ROC-AUC'         : test.get('roc_auc',  '-'),
            'PR-AUC'          : test.get('pr_auc',   '-'),
            'Hin. macro-F1'   : res.get('hinglish_macro_f1', '-'),
            'Hin. hate-F1'    : res.get('hinglish_hate_f1',  '-'),
        })

    df_ab = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("  ABLATION TABLE")
    print("=" * 100)
    print(df_ab.to_string(index=False))
    out = RESULTS_DIR / 'ablation_table.csv'
    df_ab.to_csv(out, index=False)
    print(f"\n  ✓ Ablation table → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════

def _serializable(d: dict) -> dict:
    """Remove non-JSON-serialisable keys (y_pred list, report dict)."""
    skip = {'report', 'y_pred'}
    return {k: v for k, v in d.items() if k not in skip}


def _save_metrics(all_results: dict):
    out = METRICS_OUT
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ metrics.json → {out}")

    # human-readable evaluation report
    lines = ["HATE SPEECH DETECTION — EVALUATION REPORT",
             "=" * 72, ""]
    for tag, res in all_results.items():
        lines.append(f"Model: {tag}")
        lines.append("-" * 50)
        test = res.get('test', {})
        for k, v in test.items():
            lines.append(f"  {k:20s}: {v}")
        if 'cv_macro_f1_mean' in res:
            lines.append(f"  {'CV macro-F1':20s}: "
                         f"{res['cv_macro_f1_mean']:.4f} ± {res['cv_macro_f1_std']:.4f}")
        if 'hinglish_macro_f1' in res:
            lines.append(f"  {'Hin. macro-F1':20s}: {res['hinglish_macro_f1']:.4f}")
        lines.append("")

    with open(REPORT_OUT, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ evaluation_report.txt → {REPORT_OUT}")


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Hate Speech Detection — research training pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── which model(s) ────────────────────────────────────────────────────
    p.add_argument('--model', choices=['baseline', 'transformer', 'context', 'all'],
                   default='transformer',
                   help='Which model(s) to train')
    p.add_argument('--tfidf-char', action='store_true',
                   help='Add char n-gram features to baseline TF-IDF')

    # ── data ──────────────────────────────────────────────────────────────
    p.add_argument('--data',
                   default='data/raw/merged_hate_speech_training.csv',
                   help='Path to CSV dataset')
    p.add_argument('--test-size', type=float, default=0.15,
                   help='Holdout test fraction')
    p.add_argument('--cv-folds', type=int, default=5,
                   help='Stratified k-fold folds (1 = no CV)')

    # ── loss ──────────────────────────────────────────────────────────────
    p.add_argument('--loss', choices=['ce', 'focal'], default='focal',
                   help='Loss function: class-weighted CE or Focal Loss')
    p.add_argument('--focal-gamma', type=float, default=2.0,
                   help='Focal Loss gamma (focusing parameter)')
    p.add_argument('--focal-alpha', type=float, default=0.25,
                   help='Focal Loss alpha (minority class weight)')

    # ── model architecture ────────────────────────────────────────────────
    p.add_argument('--base-model', default='xlm-roberta-base',
                   help='HuggingFace model name or local path')
    p.add_argument('--max-length', type=int, default=128,
                   help='Tokenizer max sequence length')
    p.add_argument('--dropout', type=float, default=0.1,
                   help='Dropout probability for transformer')

    # ── training hyperparams ──────────────────────────────────────────────
    p.add_argument('--epochs', type=int, default=3,
                   help='Max training epochs')
    p.add_argument('--batch-size', type=int, default=16,
                   help='Training batch size')
    p.add_argument('--lr', type=float, default=2e-5,
                   help='AdamW learning rate')
    p.add_argument('--weight-decay', type=float, default=0.01,
                   help='AdamW weight decay')
    p.add_argument('--scheduler', choices=['linear', 'cosine'], default='linear',
                   help='LR scheduler type')
    p.add_argument('--early-stopping-patience', type=int, default=2,
                   help='Epochs without val macro-F1 improvement before stop')

    # ── inference ─────────────────────────────────────────────────────────
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Decision threshold (0.5 = auto-tune from val sweep)')

    # ── reproducibility ───────────────────────────────────────────────────
    p.add_argument('--seed', type=int, default=42, help='Random seed')

    return p


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # seed everything
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch; torch.manual_seed(args.seed)
    except ImportError:
        pass

    _make_dirs()

    data_path = PROJECT_ROOT / args.data
    df = _load_dataset(data_path)
    df_tv, df_test = _holdout_split(df, test_size=args.test_size, seed=args.seed)

    all_results: dict = {}

    run_baseline    = args.model in ('baseline',    'all')
    run_transformer = args.model in ('transformer', 'all')
    run_context     = args.model in ('context',     'all')

    if run_baseline:
        try:
            all_results['baseline'] = train_baseline_model(df_tv, df_test, args)
        except Exception as exc:
            print(f"\n✗ Baseline training failed: {exc}")

    if run_transformer:
        try:
            all_results[f'transformer_{args.loss}'] = train_transformer_model(
                df_tv, df_test, args
            )
        except Exception as exc:
            print(f"\n✗ Transformer training failed: {exc}")

    if run_context:
        try:
            all_results[f'context_{args.loss}'] = train_context_model(
                df_tv, df_test, args
            )
        except Exception as exc:
            print(f"\n✗ Context model training failed: {exc}")

    if all_results:
        _print_ablation_table(all_results)
        _save_metrics(all_results)

    print("\n" + "=" * 64)
    print("  PIPELINE COMPLETE")
    print("=" * 64)
    print(f"  Results  → {RESULTS_DIR}")
    print(f"  Models   → {MODELS_DIR}")
    print(f"  Metrics  → {METRICS_OUT}")
    print(f"  Report   → {REPORT_OUT}\n")


if __name__ == '__main__':
    main()
