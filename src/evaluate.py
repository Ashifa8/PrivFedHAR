"""
evaluate.py
===========
Evaluation utilities for PrivFedHAR:
  - compute_all_metrics (accuracy, precision, recall, F1, AUC, specificity,
                         Brier score, ECE)
  - safe_auc (handles single-class folds — returns 0.5 instead of crashing)
  - evaluate_model (run inference on test set, return full metrics)
  - generalization_test (held-out subject evaluation)
  - plot_results (loss curves, confusion matrix, ROC curves)

Authors: Ashifa Ikram, Shanzae Khan, Atif Saeed
         FAST NUCES Islamabad
"""

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc as sk_auc
)
from sklearn.preprocessing import label_binarize


# ─── Safe AUC (handles single-class folds) ───────────────────────────────────
def safe_auc(y_true: np.ndarray,
             y_prob: np.ndarray,
             n_classes: int) -> float:
    """
    Compute macro OvR AUC safely.

    For S105, S108, S109 in PAMAP2 — these subjects have only a single
    activity class in their LOSO test fold.  One-vs-rest AUC is undefined
    for single-class inputs, so we return 0.50 (random-chance baseline)
    rather than raising an exception.

    Returns:
        AUC ∈ [0, 1], or 0.5 if fewer than 2 classes present.
    """
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.5   # dataset constraint, not model failure

    try:
        classes = sorted(unique.tolist())
        y_bin   = label_binarize(y_true, classes=list(range(n_classes)))
        if len(classes) == 2:
            return float(roc_auc_score(y_bin[:, classes[1]], y_prob[:, classes[1]]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(roc_auc_score(
                y_bin[:, classes], y_prob[:, classes],
                average='macro', multi_class='ovr'))
    except Exception as e:
        print(f"  [AUC WARNING] {e}. Returning 0.5.")
        return 0.5


# ─── ECE ─────────────────────────────────────────────────────────────────────
def compute_ece(y_true: np.ndarray,
                y_prob: np.ndarray,
                n_bins: int = 10) -> float:
    """Expected Calibration Error (lower is better)."""
    y_pred      = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    correct     = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            ece += mask.sum() / len(y_true) * abs(
                correct[mask].mean() - confidences[mask].mean())
    return float(ece)


# ─── Comprehensive metrics ────────────────────────────────────────────────────
def compute_all_metrics(y_true:     np.ndarray,
                        y_pred:     np.ndarray,
                        y_prob:     np.ndarray,
                        num_classes: int) -> dict:
    """
    Compute all PrivFedHAR evaluation metrics.

    Returns dict with:
        accuracy, precision, recall, f1_macro, auc_macro,
        specificity, brier_score, ece, confusion_matrix
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc  = safe_auc(y_true, y_prob, num_classes)

    # Confusion matrix → per-class specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    spec_per_class = []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FP - FN
        denom = TN + FP
        spec_per_class.append(TN / denom if denom > 0 else 0.0)
    specificity = float(np.mean(spec_per_class))

    # Brier score
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    brier = float(np.mean((y_prob - y_bin) ** 2))

    ece = compute_ece(y_true, y_prob)

    return {
        'accuracy':        acc,
        'precision':       prec,
        'recall':          rec,
        'f1_macro':        f1,
        'auc_macro':       auc,
        'specificity':     specificity,
        'brier_score':     brier,
        'ece':             ece,
        'confusion_matrix': cm,
    }


# ─── Model inference ─────────────────────────────────────────────────────────
def evaluate_model(model,
                   X_test:      np.ndarray,
                   y_test:      np.ndarray,
                   device:      torch.device,
                   num_classes: int,
                   batch_size:  int = 128):
    """
    Run inference on test set and compute all metrics.

    Returns:
        metrics:  dict of all evaluation metrics
        y_true:   ground-truth labels
        y_pred:   predicted class indices
        y_prob:   softmax probabilities (n, num_classes)
    """
    model.eval()
    all_preds, all_probs, all_true = [], [], []

    dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for X_b, y_b in dl:
            logits, _ = model(X_b.to(device))
            probs     = F.softmax(logits, dim=-1).cpu().numpy()
            preds     = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_b.numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Pad probability matrix if model saw fewer classes
    if y_prob.shape[1] < num_classes:
        full = np.zeros((len(y_true), num_classes))
        full[:, :y_prob.shape[1]] = y_prob
        y_prob = full

    metrics = compute_all_metrics(y_true, y_pred, y_prob, num_classes)
    return metrics, y_true, y_pred, y_prob


# ─── Plotting ─────────────────────────────────────────────────────────────────
def plot_fl_loss_curves(loso_results: list[dict],
                        save_path: str = 'results/figures/fl_loss_curves.png'):
    """Plot per-subject FL training loss curves across rounds."""
    import matplotlib.pyplot as plt
    os_make_dir(save_path)

    n = len(loso_results)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('FL Training Loss Curves — Per Subject\n'
                 'PrivFedHAR (PAMAP2)', fontsize=13, fontweight='bold')
    axes = axes.flatten()
    cmap = plt.cm.tab10

    for i, r in enumerate(loso_results):
        ax     = axes[i]
        hist   = r['fl_history']
        rounds = [h['round'] for h in hist]
        losses = [h['loss']  for h in hist]
        ax.plot(rounds, losses, color=cmap(i / n), lw=2.5, label='Train Loss')
        ax.set_title(f'Subject {r["test_subject"]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('FL Round', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          activity_names: dict,
                          subject_id: int,
                          save_path: str = 'results/figures/confusion_matrix.png'):
    """Plot normalized confusion matrix for a single subject."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    os_make_dir(save_path)

    n_classes  = max(len(activity_names), int(y_true.max()) + 1)
    labels     = [activity_names.get(i, str(i)) for i in range(n_classes)]
    cm         = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    row_sums   = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm    = cm / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.3, cbar=False)
    ax.set_title(f'Confusion Matrix — Subject {subject_id}', fontsize=13)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")


def plot_per_subject_accuracy(loso_results: list[dict],
                              save_path: str = 'results/figures/per_subject_accuracy.png'):
    """Bar chart of per-subject accuracy and F1."""
    import matplotlib.pyplot as plt
    os_make_dir(save_path)

    subjects = [r['test_subject'] for r in loso_results]
    accs     = [r['accuracy']     for r in loso_results]
    f1s      = [r['f1_macro']     for r in loso_results]
    x        = np.arange(len(subjects))
    labels   = [f'S{s}' for s in subjects]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, accs, 0.35, label='Accuracy',  color='steelblue', alpha=0.85)
    ax.bar(x + 0.2, f1s,  0.35, label='F1 Macro',  color='coral',     alpha=0.85)
    ax.axhline(np.mean(accs), color='steelblue', ls='--', lw=1.5,
               label=f'μ Acc = {np.mean(accs):.3f}')
    ax.axhline(np.mean(f1s), color='coral', ls='--', lw=1.5,
               label=f'μ F1 = {np.mean(f1s):.3f}')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 1.1)
    ax.set_title('Per-Subject LOSO Performance — PrivFedHAR', fontsize=12)
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")


def print_loso_summary(loso_results: list[dict]):
    """Print a formatted LOSO results table."""
    METRICS = ['accuracy', 'precision', 'recall', 'f1_macro',
               'auc_macro', 'specificity', 'brier_score', 'ece']
    LABELS  = {
        'accuracy':    'Accuracy',     'precision': 'Precision (Macro)',
        'recall':      'Recall (Macro)', 'f1_macro': 'F1-Score (Macro)',
        'auc_macro':   'AUC-ROC',      'specificity': 'Specificity',
        'brier_score': 'Brier Score',   'ece':       'ECE',
    }
    print('\n' + '='*72)
    print('PRIVFEDHAR — LOSO RESULTS  (PAMAP2, 9 subjects, 15 activities)')
    print('='*72)
    print(f'{"Subject":>9} {"Acc":>8} {"Prec":>8} {"Rec":>8} '
          f'{"F1":>8} {"AUC":>8} {"Spec":>8} {"N":>6}')
    print('-'*72)
    for r in sorted(loso_results, key=lambda x: x['test_subject']):
        print(f'  Sub {r["test_subject"]:2d}   '
              f'{r["accuracy"]:>8.4f} {r["precision"]:>8.4f} '
              f'{r["recall"]:>8.4f} {r["f1_macro"]:>8.4f} '
              f'{r["auc_macro"]:>8.4f} {r["specificity"]:>8.4f} '
              f'{r["test_samples"]:>6}')
    print('-'*72)
    for k in METRICS:
        vals = [r[k] for r in loso_results]
        print(f'  {LABELS[k]:<25}: '
              f'{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}  '
              f'[{np.nanmin(vals):.4f}, {np.nanmax(vals):.4f}]')
    print('='*72)


# ─── Helper ───────────────────────────────────────────────────────────────────
def os_make_dir(filepath: str):
    import os
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
