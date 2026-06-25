"""
Proposed Model Package for EEG Analysis.
Exposes feature extraction methods, classical machine learning models,
deep learning baselines, EEG-specific architectures, and multi-task/DANN models.
"""

from .features import (
    extract_psd_features,
    extract_differential_entropy,
    extract_faa_features,
    extract_all_features
)
from .classical_models import (
    get_svm_pipeline,
    get_random_forest_pipeline,
    get_xgboost_pipeline
)
from .deep_learning_baselines import (
    build_1d_cnn,
    build_lstm,
    build_cnn_lstm
)
from .eeg_models import (
    EEGNet,
    TSception,
    EEGTransformer
)
from .multitask_dann_models import (
    GradientReversalLayer,
    build_multitask_eeg_model,
    build_dann_eeg_model,
    masked_binary_crossentropy
)

__all__ = [
    'extract_psd_features',
    'extract_differential_entropy',
    'extract_faa_features',
    'extract_all_features',
    'get_svm_pipeline',
    'get_random_forest_pipeline',
    'get_xgboost_pipeline',
    'build_1d_cnn',
    'build_lstm',
    'build_cnn_lstm',
    'EEGNet',
    'TSception',
    'EEGTransformer',
    'GradientReversalLayer',
    'build_multitask_eeg_model',
    'build_dann_eeg_model',
    'masked_binary_crossentropy'
]
