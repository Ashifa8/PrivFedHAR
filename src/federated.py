"""
federated.py
============
Federated Learning training loop for PrivFedHAR.

Implements:
  - FedAvg aggregation (shared backbone only — SALN excluded)
  - ClientTrainer (local training with DP-SGD via Opacus)
  - personalize_for_subject (local fine-tuning of SALN + classifier head)
  - run_loso_fold (single LOSO fold: FL + personalization + evaluation)
  - run_loso_evaluation (full 9-subject LOSO evaluation)

FedAvg equation:
  θ^(r+1) = Σ_k (n_k / N) · θ_k^r

Authors: Ashifa Ikram, Shanzae Khan, Atif Saeed
         FAST NUCES Islamabad
"""

import os
import time
import copy
import pickle
import gc
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .model import PersonalizedFedLSTMGRU
from .preprocess import loso_split, NUM_FEATURES
from .evaluate import compute_all_metrics, evaluate_model

# ─── Defaults ────────────────────────────────────────────────────────────────
FL_ROUNDS        = 30
LOCAL_EPOCHS     = 5
FINE_TUNE_EPOCHS = 20
BATCH_SIZE       = 128
LR_GLOBAL        = 1e-3
LR_FINETUNE      = 5e-4
HIDDEN_DIM       = 128
DROPOUT          = 0.4
N_HEADS_ATTN     = 4
PATIENCE         = 12
MAX_EPOCHS       = 50
SEED             = 42

DP_ENABLED      = True
DP_MAX_GRAD_NORM = 1.0
DP_NOISE_MULT   = 0.8
DP_DELTA        = 1e-5


# ─── Opacus import (optional) ────────────────────────────────────────────────
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("[INFO] Opacus not available — DP disabled. Install with: pip install opacus")


def make_dp_compatible(model):
    if OPACUS_AVAILABLE:
        return ModuleValidator.fix(model)
    return model


# ─── FedAvg ──────────────────────────────────────────────────────────────────
def federated_average(client_states: list[dict],
                      client_sizes:  list[int]) -> dict:
    """
    Weighted FedAvg over shared backbone parameters.
    SALN parameters (subject_gamma, subject_beta) are excluded.

    θ^(r+1) = Σ_k (n_k / N) · θ_k^r
    """
    total   = sum(client_sizes)
    weights = [s / total for s in client_sizes]
    agg: dict = {}
    for key in client_states[0].keys():
        if 'subject_gamma' in key or 'subject_beta' in key:
            continue   # SALN stays local
        stacked = torch.stack([
            client_states[i][key].float() * weights[i]
            for i in range(len(client_states))
        ])
        agg[key] = stacked.sum(dim=0)
    return agg


def broadcast_global_to_client(global_state: dict,
                                client_model: PersonalizedFedLSTMGRU):
    """Copy global shared params into client model, preserving SALN params."""
    local_state = client_model.state_dict()
    for key, val in global_state.items():
        if key in local_state:
            local_state[key] = val.to(local_state[key].device)
    client_model.load_state_dict(local_state)


# ─── DataLoader builder ───────────────────────────────────────────────────────
def build_dataloader(X: np.ndarray, y: np.ndarray,
                     batch_size: int = BATCH_SIZE,
                     shuffle: bool = True,
                     balanced: bool = True) -> DataLoader:
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    ds  = TensorDataset(X_t, y_t)
    if shuffle and balanced:
        counts  = np.bincount(y, minlength=int(y.max()) + 1).astype(float)
        counts  = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts[y]
        sampler = WeightedRandomSampler(torch.FloatTensor(weights), len(weights))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ─── Early Stopping ───────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter   = 0

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None or self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─── ClientTrainer ────────────────────────────────────────────────────────────
class ClientTrainer:
    """
    Manages local training for a single federated client (subject).

    Applies DP-SGD (Opacus) with:
        σ = 0.8  (noise multiplier)
        C = 1.0  (max gradient norm)
        δ = 1e-5
    """
    def __init__(self,
                 subject_id: int,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val:   np.ndarray,
                 y_val:   np.ndarray,
                 num_classes: int,
                 device: torch.device,
                 lr: float = LR_GLOBAL,
                 use_dp: bool = DP_ENABLED):
        self.subject_id  = subject_id
        self.device      = device
        self.X_train     = X_train
        self.y_train     = y_train
        self.X_val       = X_val
        self.y_val       = y_val
        self.n_samples   = len(X_train)
        self.num_classes = num_classes
        self.use_dp      = use_dp and OPACUS_AVAILABLE
        self.lr          = lr
        self.privacy_engine = None
        self.early_stopping = EarlyStopping(patience=PATIENCE)

    def setup_model(self, global_state: Optional[dict] = None):
        self.model = PersonalizedFedLSTMGRU(
            NUM_FEATURES, HIDDEN_DIM, self.num_classes,
            subject_id=self.subject_id, dropout=DROPOUT
        ).to(self.device)

        if global_state is not None:
            broadcast_global_to_client(global_state, self.model)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        if self.use_dp:
            try:
                self.model = make_dp_compatible(self.model)
                dl = build_dataloader(self.X_train, self.y_train,
                                      batch_size=BATCH_SIZE, balanced=False)
                pe = PrivacyEngine()
                self.model, self.optimizer, self.train_loader = pe.make_private(
                    module=self.model, optimizer=self.optimizer,
                    data_loader=dl,
                    noise_multiplier=DP_NOISE_MULT,
                    max_grad_norm=DP_MAX_GRAD_NORM,
                )
                self.privacy_engine = pe
            except Exception as e:
                print(f"  [S{self.subject_id}] DP setup failed: {e}. Falling back to no-DP.")
                self.use_dp = False
                self.train_loader = build_dataloader(self.X_train, self.y_train)
        else:
            self.train_loader = build_dataloader(self.X_train, self.y_train)

        self.val_loader = build_dataloader(
            self.X_val, self.y_val, shuffle=False, balanced=False
        ) if len(self.X_val) > 0 else None

    def _validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in self.val_loader:
                X_b = X_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)
                logits, _ = self.model(X_b)
                total_loss += self.criterion(logits, y_b).item() * len(y_b)
                total_n    += len(y_b)
        self.model.train()
        return total_loss / max(total_n, 1)

    def train_one_round(self, n_epochs: int = LOCAL_EPOCHS):
        """Run n_epochs of local DP-SGD training."""
        self.model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for epoch in range(n_epochs):
            self.model.train()
            for X_b, y_b in self.train_loader:
                X_b = X_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                logits, _ = self.model(X_b)
                loss = self.criterion(logits, y_b)
                loss.backward()
                if not self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), DP_MAX_GRAD_NORM)
                self.optimizer.step()
                total_loss    += loss.item() * len(y_b)
                total_correct += (logits.argmax(1) == y_b).sum().item()
                total_n       += len(y_b)

            val_loss = self._validate()
            if self.early_stopping(val_loss):
                break

        eps = None
        if self.use_dp and self.privacy_engine is not None:
            try:
                eps = self.privacy_engine.get_epsilon(delta=DP_DELTA)
            except Exception:
                pass

        state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        return (state,
                total_loss / max(total_n, 1),
                total_correct / max(total_n, 1),
                eps)


# ─── Local fine-tuning ────────────────────────────────────────────────────────
def personalize_for_subject(global_state: dict,
                             X_support: np.ndarray,
                             y_support: np.ndarray,
                             subject_id: int,
                             num_classes: int,
                             device: torch.device,
                             n_epochs: int = FINE_TUNE_EPOCHS,
                             lr: float = LR_FINETUNE) -> PersonalizedFedLSTMGRU:
    """
    Fine-tune SALN parameters and classifier head on the test subject's
    support set (30% of test windows). Backbone parameters are frozen.

    Achieves subject-specific adaptation without raw data leaving the device.
    """
    model = PersonalizedFedLSTMGRU(
        NUM_FEATURES, HIDDEN_DIM, num_classes,
        subject_id=subject_id, dropout=DROPOUT
    ).to(device)
    broadcast_global_to_client(global_state, model)

    # Freeze backbone — only SALN + classifier are trainable
    for name, param in model.named_parameters():
        param.requires_grad = any(
            k in name for k in ['subject_gamma', 'subject_beta', 'classifier'])

    if len(X_support) < 5:
        return model

    try:
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_support, y_support, test_size=0.2,
            stratify=y_support, random_state=SEED)
    except ValueError:
        X_tr, X_v, y_tr, y_v = X_support, X_support, y_support, y_support

    bs      = min(32, len(X_tr))
    dl_tr   = build_dataloader(X_tr, y_tr, batch_size=bs)
    dl_v    = build_dataloader(X_v, y_v, batch_size=bs, shuffle=False, balanced=False)
    opt     = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, weight_decay=1e-4)
    crit    = nn.CrossEntropyLoss(label_smoothing=0.05)
    es      = EarlyStopping(patience=min(5, n_epochs // 4))

    for epoch in range(n_epochs):
        model.train()
        for X_b, y_b in dl_tr:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            opt.zero_grad()
            logits, _ = model(X_b)
            crit(logits, y_b).backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in dl_v:
                logits, _ = model(X_b.to(device))
                val_loss += crit(logits, y_b.to(device)).item()
        if es(val_loss / max(len(dl_v), 1)):
            break

    return model


# ─── Single LOSO fold ─────────────────────────────────────────────────────────
def run_loso_fold(encoded_raw: dict,
                  test_subject: int,
                  num_classes: int,
                  device: torch.device,
                  checkpoint_dir: str = './checkpoints',
                  verbose: bool = True) -> dict:
    """
    Run one complete LOSO fold:
      1. Split data (train clients / test subject)
      2. Federated training (FL_ROUNDS rounds of FedAvg + DP-SGD)
      3. Local fine-tuning (SALN + classifier on support set)
      4. Evaluation on held-out test windows

    Returns a result dict with all metrics and FL history.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    t0 = time.time()

    if verbose:
        print(f"\n{'='*60}\n[S{test_subject}] LOSO fold — test subject: {test_subject}\n{'='*60}")

    client_data, X_test, y_test, _ = loso_split(encoded_raw, test_subject)

    # Initialize global model
    global_model = PersonalizedFedLSTMGRU(
        NUM_FEATURES, HIDDEN_DIM, num_classes, subject_id=0, dropout=DROPOUT)
    global_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
    del global_model

    # Setup clients
    clients: dict[int, ClientTrainer] = {}
    for sid, (Xtr, ytr, Xv, yv) in client_data.items():
        ct = ClientTrainer(sid, Xtr, ytr, Xv, yv, num_classes, device)
        ct.setup_model(global_state)
        clients[sid] = ct

    # ── Federated training ────────────────────────────────────────────────
    fl_history = []
    fl_bar = tqdm(range(FL_ROUNDS), desc=f'[S{test_subject}] FL',
                  ncols=90, colour='blue', leave=True)

    for fl_round in fl_bar:
        round_states, round_sizes, round_losses, round_accs = [], [], [], []

        for sid, ct in clients.items():
            broadcast_global_to_client(global_state, ct.model)
            state, loss, acc, eps = ct.train_one_round(LOCAL_EPOCHS)
            round_states.append(state)
            round_sizes.append(ct.n_samples)
            round_losses.append(loss)
            round_accs.append(acc)

        agg = federated_average(round_states, round_sizes)
        global_state.update(agg)

        avg_loss = float(np.mean(round_losses))
        avg_acc  = float(np.mean(round_accs))
        fl_history.append({'round': fl_round + 1, 'loss': avg_loss, 'acc': avg_acc})
        fl_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.4f}'})

    # ── Personalization ───────────────────────────────────────────────────
    n_support = max(10, int(0.30 * len(X_test)))
    idx       = np.random.permutation(len(X_test))
    sup_idx   = idx[:n_support]
    eval_idx  = idx[n_support:] if len(idx) > n_support else idx

    personalized_model = personalize_for_subject(
        global_state, X_test[sup_idx], y_test[sup_idx],
        subject_id=test_subject, num_classes=num_classes, device=device)

    # ── Evaluation ────────────────────────────────────────────────────────
    metrics, y_true, y_pred, y_prob = evaluate_model(
        personalized_model, X_test[eval_idx], y_test[eval_idx],
        device, num_classes, batch_size=BATCH_SIZE)

    elapsed = time.time() - t0
    if verbose:
        print(f"[S{test_subject}] Acc={metrics['accuracy']:.4f} | "
              f"F1={metrics['f1_macro']:.4f} | AUC={metrics['auc_macro']:.4f} | "
              f"Time={elapsed:.1f}s")

    # Cleanup GPU memory
    del clients
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'test_subject': test_subject,
        'test_samples': len(y_test[eval_idx]),
        'fl_history':   fl_history,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,
        **metrics
    }


# ─── Full LOSO evaluation ────────────────────────────────────────────────────
def run_loso_evaluation(encoded_raw: dict,
                        num_classes: int = 15,
                        device: Optional[torch.device] = None,
                        checkpoint_dir: str = './checkpoints') -> list[dict]:
    """
    Run the complete 9-subject LOSO evaluation.

    Args:
        encoded_raw:     {subject_id: (X, y)} from preprocess.py
        num_classes:     number of activity classes (default 15)
        device:          torch device (auto-detects GPU if None)
        checkpoint_dir:  directory to save per-fold checkpoints

    Returns:
        List of result dicts, one per subject fold.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    subjects     = sorted(encoded_raw.keys())
    loso_results = []

    outer_bar = tqdm(subjects, desc='LOSO Folds', unit='subject',
                     ncols=100, colour='green')

    for sid in outer_bar:
        result = run_loso_fold(encoded_raw, sid, num_classes, device,
                               checkpoint_dir, verbose=True)
        loso_results.append(result)
        outer_bar.set_postfix({
            'last': f'S{sid}',
            'acc':  f'{result["accuracy"]:.4f}',
            'f1':   f'{result["f1_macro"]:.4f}'
        })

    # Print summary
    accs = [r['accuracy']  for r in loso_results]
    f1s  = [r['f1_macro']  for r in loso_results]
    aucs = [r['auc_macro'] for r in loso_results]
    print(f"\n{'='*60}")
    print(f"LOSO COMPLETE — {len(loso_results)} subjects")
    print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1 Macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC      : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"{'='*60}")

    return loso_results
