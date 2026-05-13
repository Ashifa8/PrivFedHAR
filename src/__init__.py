"""
PrivFedHAR — Privacy-Preserving Personalized Federated Learning
for Healthcare Activity Recognition

Authors: Ashifa Ikram, Shanzae Khan, Atif Saeed
         FAST NUCES Islamabad
"""

from .preprocess import load_and_preprocess, loso_split
from .model      import PersonalizedFedLSTMGRU, SubjectAdaptiveLayerNorm
from .federated  import run_loso_evaluation, run_loso_fold
from .evaluate   import evaluate_model, print_loso_summary

__version__ = "1.0.0"
__all__ = [
    "load_and_preprocess", "loso_split",
    "PersonalizedFedLSTMGRU", "SubjectAdaptiveLayerNorm",
    "run_loso_evaluation", "run_loso_fold",
    "evaluate_model", "print_loso_summary",
]
