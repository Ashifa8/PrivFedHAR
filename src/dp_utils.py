"""
dp_utils.py
===========
Differential Privacy utilities for PrivFedHAR.

Implements (ε, δ)-DP via Opacus (per-sample gradient clipping + Gaussian noise):
    g̃ = (1/B)[Σᵢ clip(gᵢ, C) + N(0, σ²C²I)]

with σ=0.8, C=1.0, δ=10⁻⁵.

Privacy accounting is handled by Opacus moments accountant internally.

Authors: Ashifa Ikram, Shanzae Khan, Atif Saeed
         FAST NUCES Islamabad
"""

# DP hyperparameters
DP_NOISE_MULT    = 0.8     # Gaussian noise multiplier σ
DP_MAX_GRAD_NORM = 1.0     # Per-sample gradient clipping norm C
DP_DELTA         = 1e-5    # δ in (ε, δ)-DP guarantee


def get_dp_config() -> dict:
    """Return the DP configuration used in PrivFedHAR."""
    return {
        'noise_multiplier': DP_NOISE_MULT,
        'max_grad_norm':    DP_MAX_GRAD_NORM,
        'delta':            DP_DELTA,
    }


def try_import_opacus():
    """
    Safely import Opacus.
    Returns (PrivacyEngine, ModuleValidator) or (None, None) if unavailable.
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        return PrivacyEngine, ModuleValidator
    except ImportError:
        print("[DP] Opacus not available. Install: pip install opacus")
        return None, None


def make_model_dp_compatible(model):
    """
    Apply Opacus ModuleValidator fixes to make a model DP-compatible.
    Replaces BatchNorm layers with GroupNorm (which supports per-sample gradients).
    """
    PrivacyEngine, ModuleValidator = try_import_opacus()
    if ModuleValidator is not None:
        return ModuleValidator.fix(model)
    return model


def attach_privacy_engine(model, optimizer, data_loader):
    """
    Attach Opacus PrivacyEngine to a model and optimizer.

    Args:
        model:       nn.Module (already made DP-compatible)
        optimizer:   torch optimizer
        data_loader: training DataLoader

    Returns:
        (dp_model, dp_optimizer, dp_loader, privacy_engine)
        or original (model, optimizer, data_loader, None) if Opacus unavailable.
    """
    PrivacyEngine, _ = try_import_opacus()
    if PrivacyEngine is None:
        return model, optimizer, data_loader, None

    pe = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=DP_NOISE_MULT,
        max_grad_norm=DP_MAX_GRAD_NORM,
    )
    print(f"[DP] Privacy engine attached | σ={DP_NOISE_MULT} | C={DP_MAX_GRAD_NORM} | δ={DP_DELTA}")
    return dp_model, dp_optimizer, dp_loader, pe


def get_epsilon(privacy_engine, delta: float = DP_DELTA) -> float | None:
    """
    Query the current privacy budget ε from the Opacus moments accountant.

    Args:
        privacy_engine: Opacus PrivacyEngine object (or None)
        delta:          δ parameter

    Returns:
        ε (float) or None if unavailable
    """
    if privacy_engine is None:
        return None
    try:
        return float(privacy_engine.get_epsilon(delta=delta))
    except Exception as e:
        print(f"[DP] Could not compute ε: {e}")
        return None


def clip_gradients(model, max_norm: float = DP_MAX_GRAD_NORM):
    """
    Manual gradient clipping (used when Opacus is not available).
    Clips L2 norm of all gradients to max_norm.
    """
    import torch.nn as nn
    import torch
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def privacy_report(privacy_engine, delta: float = DP_DELTA):
    """Print a summary of the current privacy expenditure."""
    eps = get_epsilon(privacy_engine, delta)
    if eps is not None:
        print(f"\n{'='*40}")
        print(f"Privacy Report:")
        print(f"  ε = {eps:.4f}")
        print(f"  δ = {delta:.2e}")
        print(f"  σ = {DP_NOISE_MULT}")
        print(f"  C = {DP_MAX_GRAD_NORM}")
        print(f"  Guarantee: ({eps:.2f}, {delta:.2e})-DP")
        print(f"{'='*40}\n")
    else:
        print("[DP] Privacy engine not available — no privacy guarantee.")
