import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
import gc

# Add src/ folder to sys.path to enable importing proposed_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, balanced_accuracy_score)

# Import proposed models
from proposed_model import (
    extract_all_features,
    get_svm_pipeline,
    get_random_forest_pipeline,
    get_xgboost_pipeline,
    build_1d_cnn,
    build_lstm,
    build_cnn_lstm,
    EEGNet,
    TSception,
    EEGTransformer,
    build_multitask_eeg_model,
    build_dann_eeg_model,
    masked_binary_crossentropy
)

# Constants resolved relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
FEATURES_CACHE = os.path.join(DATA_DIR, "extracted_features_tuned.npy")
REPORT_PATH = os.path.join(PROJECT_ROOT, "report.md")
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 5e-4  # Tuned learning rate for stable convergence

def load_data():
    print("Loading combined EEG data files...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_depression = np.load(os.path.join(DATA_DIR, "y_depression.npy"))
    y_stress = np.load(os.path.join(DATA_DIR, "y_stress.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"), allow_pickle=True)
    datasets = np.load(os.path.join(DATA_DIR, "datasets.npy"))
    
    print(f"Loaded: X {X.shape}, y_depression {y_depression.shape}, y_stress {y_stress.shape}")
    return X, y_depression, y_stress, subjects, datasets

def get_or_extract_features(X):
    if os.path.exists(FEATURES_CACHE):
        print("Loading cached classical features...")
        features = np.load(FEATURES_CACHE)
    else:
        print("Extracting classical features (Relative PSD, DE, FAA, Spectral Entropy, RMS, STD) for all segments. This may take 1-2 minutes...")
        start_time = time.time()
        features = extract_all_features(X)
        np.save(FEATURES_CACHE, features)
        print(f"Feature extraction complete in {time.time() - start_time:.2f} seconds. Cached to {FEATURES_CACHE}.")
    print(f"Features matrix shape: {features.shape}")
    return features

def calculate_metrics(y_true, y_pred, y_prob, threshold=0.5):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = roc_auc_score(y_true, y_pred) # fallback if AUC fails

    # Accuracy of always predicting the most frequent class. Any model scoring
    # below this is worse than a constant predictor.
    counts = np.bincount(np.asarray(y_true).astype(int), minlength=2)
    majority = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0

    return {
        "Accuracy": float(acc),
        "Balanced Acc": float(balanced_accuracy_score(y_true, y_pred)),
        "Majority Baseline": majority,
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(auc),
        "Threshold": float(threshold)
    }

def tune_threshold(y_true, y_prob):
    """
    Picks the decision threshold maximizing balanced accuracy on a held-out
    validation split. A fixed 0.5 cutoff is arbitrary under class imbalance.

    Balanced accuracy rather than F1: F1 ignores true negatives, so maximizing it
    under imbalance degenerates to predicting every sample positive (which scores
    2p/(1+p) for base rate p and beats any real cutoff on a weak classifier).
    Balanced accuracy scores an all-positive prediction at exactly 0.5, so a
    degenerate threshold can only win when there is genuinely no signal.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5
    # Search candidate cutoffs drawn from the score distribution itself, so the
    # grid adapts to models whose probabilities are squashed into a narrow band.
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    best_t, best_score = None, -1.0
    for t in candidates:
        pred = (y_prob >= t).astype(int)
        # Skip degenerate cutoffs. All-positive AND all-negative both score exactly
        # 0.5 balanced accuracy, so on a weak/small validation split one of them can
        # win and discard an otherwise good ranking (observed: a model with test
        # AUC 0.946 pinned to 0.5 balanced accuracy by a threshold of 0.98).
        if pred.min() == pred.max():
            continue
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score, best_t = score, t
    if best_t is None:
        return float(np.median(y_prob))
    return float(best_t)

DATASET_NAMES = {0: "MODMA", 1: "SAM40", 2: "CogLoad", 3: "AMIGOS"}


def per_dataset_auc(y_test, prob_test, ds_test):
    """
    ROC-AUC split by source corpus.

    A pooled stress figure is dominated ~78% by AMIGOS, whose arousal label is not
    EEG-decodable (Phase 1: 0.4875 within-subject AUC, unchanged by relabelling).
    Reporting only the pooled number hides that SAM40 is around 0.72. Always read
    the breakdown, not the pooled row.
    """
    out = {}
    for d in np.unique(ds_test):
        m = ds_test == d
        if m.sum() < 2 or len(np.unique(y_test[m])) < 2:
            continue
        out[DATASET_NAMES.get(int(d), str(d))] = float(roc_auc_score(y_test[m], prob_test[m]))
    return out


def subject_level_metrics(y_test, prob_test, subj_test):
    """
    Aggregate segment posteriors to one score per subject.

    For depression every segment of a subject carries the same label, so the
    meaningful question is whether the *subject* is MDD, not whether a 4-second
    window is. Segment-level metrics on 50%-overlapping windows overstate this.
    """
    subs = np.unique(subj_test)
    if len(subs) < 2:
        return {}
    s_true, s_prob = [], []
    for s in subs:
        m = subj_test == s
        vals = np.unique(y_test[m])
        if len(vals) != 1:      # subject spans both classes -> not a subject-level task
            return {}
        s_true.append(int(vals[0]))
        s_prob.append(float(prob_test[m].mean()))
    s_true, s_prob = np.array(s_true), np.array(s_prob)
    if len(np.unique(s_true)) < 2:
        return {}
    thr = np.quantile(s_prob, 1.0 - s_true.mean())
    return {
        "n_subjects": int(len(subs)),
        "auc": float(roc_auc_score(s_true, s_prob)),
        "balanced_acc": float(balanced_accuracy_score(s_true, (s_prob >= thr).astype(int))),
    }


def score_with_tuned_threshold(y_val, prob_val, y_test, prob_test, elapsed,
                               ds_test=None, subj_test=None):
    """
    Tunes the threshold on validation, then transfers it to test as a *quantile*
    rather than as an absolute cutoff.

    Models emit systematically different probability ranges for different subjects,
    so an absolute cutoff tuned on only 5 validation subjects can place every test
    subject on one side of it — observed as balanced accuracy pinned to exactly
    0.5000 on models whose ROC-AUC was 0.98. The cutoff's rank position survives
    that shift; its raw value does not.

    Uses the test *scores* to locate the quantile, never the test labels.
    """
    prob_val = np.asarray(prob_val).ravel()
    prob_test = np.asarray(prob_test).ravel()

    t_val = tune_threshold(y_val, prob_val)
    # Fraction of validation predicted positive at the tuned cutoff.
    pos_rate = float((prob_val >= t_val).mean())
    # Keep at least one sample on each side so the prediction is never single-class.
    lo = 1.0 / len(prob_test)
    pos_rate = min(max(pos_rate, lo), 1.0 - lo)

    t_test = float(np.quantile(prob_test, 1.0 - pos_rate))
    preds = (prob_test >= t_test).astype(int)
    out = {**calculate_metrics(y_test, preds, prob_test, threshold=t_test), "Time": elapsed}
    if ds_test is not None:
        out["PerDataset"] = per_dataset_auc(np.asarray(y_test), prob_test, np.asarray(ds_test))
    if subj_test is not None:
        out["SubjectLevel"] = subject_level_metrics(np.asarray(y_test), prob_test, np.asarray(subj_test))
    return out

def get_unified_split(subjects, datasets, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits subjects into Train/Val/Test stratified by dataset.
    This guarantees zero subject leakage across the three splits.

    The validation split is drawn per-dataset rather than taken as a contiguous
    tail of the array. Keras' validation_split takes the last fraction before
    shuffling, and since the combined arrays are concatenated in dataset order
    that produced a validation set consisting entirely of the final dataset.
    """
    unique_subs = np.unique(subjects)
    sub_to_ds = {}
    for s in unique_subs:
        idx = np.where(subjects == s)[0][0]
        sub_to_ds[s] = datasets[idx]

    train_subs = []
    val_subs = []
    test_subs = []

    # Sort dataset IDs and subjects for deterministic results
    for ds_id in sorted(np.unique(datasets)):
        ds_subs = sorted([s for s, d in sub_to_ds.items() if d == ds_id])
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(ds_subs)
        n_test = int(len(shuffled) * test_size)
        if n_test == 0 and len(shuffled) > 0:
            n_test = 1
        n_val = int(len(shuffled) * val_size)
        if n_val == 0 and len(shuffled) - n_test > 1:
            n_val = 1
        test_subs.extend(shuffled[:n_test])
        val_subs.extend(shuffled[n_test:n_test + n_val])
        train_subs.extend(shuffled[n_test + n_val:])

    train_mask = np.isin(subjects, train_subs)
    val_mask = np.isin(subjects, val_subs)
    test_mask = np.isin(subjects, test_subs)
    return train_mask, val_mask, test_mask

def fit_and_tune_model(clf_type, X_train, y_train, get_pipeline_func, groups=None, subsample_limit=None):
    """
    Fits and tunes a classical ML pipeline using GridSearchCV.
    Optionally downsamples the training set to prevent CPU time explosion.
    """
    if clf_type == "XGBoost":
        # Resolve class imbalance ratio for scale_pos_weight
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        ratio = float(num_neg / num_pos) if num_pos > 0 else 1.0
        pipeline = get_pipeline_func(scale_pos_weight=ratio)
    else:
        pipeline = get_pipeline_func()
    
    # Optional subsampling for slow estimators (like SVM on CPU)
    if subsample_limit and X_train.shape[0] > subsample_limit:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(X_train.shape[0], size=subsample_limit, replace=False)
        X_train_fit = X_train[idx_sub]
        y_train_fit = y_train[idx_sub]
        groups_fit = groups[idx_sub] if groups is not None else None
    else:
        X_train_fit = X_train
        y_train_fit = y_train
        groups_fit = groups

    # Define param grid based on classifier type
    if clf_type == "SVM":
        param_grid = {'svc__C': [0.1, 1.0, 10.0]}
    elif clf_type == "Random Forest":
        param_grid = {
            'randomforestclassifier__max_depth': [8, 12, 16],
            'randomforestclassifier__min_samples_split': [2, 5]
        }
    elif clf_type == "XGBoost":
        # Resolve the final step name dynamically to handle XGBoost or GradientBoosting fallback
        clf_name = pipeline.steps[-1][0]
        param_grid = {
            f'{clf_name}__max_depth': [3, 5],
            f'{clf_name}__learning_rate': [0.05, 0.1]
        }
    else:
        param_grid = {}

    # GroupKFold keeps a subject's segments from straddling the inner folds. A plain
    # cv=3 leaks subject identity into hyperparameter selection.
    if groups_fit is not None and len(np.unique(groups_fit)) >= 2:
        cv = GroupKFold(n_splits=min(3, len(np.unique(groups_fit))))
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_fit, y_train_fit, groups=groups_fit)
    else:
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_fit, y_train_fit)
    print(f"  Best parameters for {clf_type}: {grid.best_params_}")
    return grid.best_estimator_

def main():
    print("=== STARTING EEG MODEL EVALUATION PIPELINE WITH PARAMETER TUNING ===")
    
    # 1. Load data
    X, y_depression, y_stress, subjects, datasets = load_data()
    
    # Clean NaN segments (mostly in AMIGOS dataset)
    print("Checking for NaN values in raw EEG segments...")
    nan_mask = np.isnan(X).any(axis=(1, 2))
    nan_count = nan_mask.sum()
    if nan_count > 0:
        print(f"  Found {nan_count} segments with NaNs. Dropping them to ensure model compatibility.")
        valid_mask = ~nan_mask
        X = X[valid_mask]
        y_depression = y_depression[valid_mask]
        y_stress = y_stress[valid_mask]
        subjects = subjects[valid_mask]
        datasets = datasets[valid_mask]
        print(f"  Cleaned data shapes: X {X.shape}, y_depression {y_depression.shape}")
        
    features = get_or_extract_features(X)
    
    # 2. Split data: unified stratified subject split (70% Train, 10% Val, 20% Test)
    print("\nCreating Unified Stratified Subject splits (70% Train, 10% Val, 20% Test)...")
    train_mask, val_mask, test_mask = get_unified_split(subjects, datasets, test_size=0.2, val_size=0.1, random_state=42)

    # The validation split must span every dataset and share no subject with train or test,
    # otherwise early stopping is steered by one corpus.
    assert not (set(subjects[train_mask]) & set(subjects[val_mask])), "Subject leak: train/val overlap!"
    assert not (set(subjects[train_mask]) & set(subjects[test_mask])), "Subject leak: train/test overlap!"
    assert not (set(subjects[val_mask]) & set(subjects[test_mask])), "Subject leak: val/test overlap!"
    assert set(np.unique(datasets[val_mask])) == set(np.unique(datasets[train_mask])), \
        f"Validation set misses datasets: train={set(np.unique(datasets[train_mask]))}, val={set(np.unique(datasets[val_mask]))}"
    print(f"  Split subjects - Train: {len(set(subjects[train_mask]))}, "
          f"Val: {len(set(subjects[val_mask]))}, Test: {len(set(subjects[test_mask]))}")
    print(f"  Val set dataset composition: {dict(zip(*np.unique(datasets[val_mask], return_counts=True)))}")

    dep_mask = (y_depression != -1)
    train_idx_dep = train_mask & dep_mask
    val_idx_dep = val_mask & dep_mask
    test_idx_dep = test_mask & dep_mask

    str_mask = (y_stress != -1)
    train_idx_str = train_mask & str_mask
    val_idx_str = val_mask & str_mask
    test_idx_str = test_mask & str_mask

    X_train_dep, y_train_dep = X[train_idx_dep], y_depression[train_idx_dep]
    X_val_dep, y_val_dep = X[val_idx_dep], y_depression[val_idx_dep]
    X_test_dep, y_test_dep = X[test_idx_dep], y_depression[test_idx_dep]
    feat_train_dep, feat_val_dep, feat_test_dep = features[train_idx_dep], features[val_idx_dep], features[test_idx_dep]
    groups_train_dep = subjects[train_idx_dep]
    subj_test_dep = subjects[test_idx_dep]

    X_train_str, y_train_str = X[train_idx_str], y_stress[train_idx_str]
    X_val_str, y_val_str = X[val_idx_str], y_stress[val_idx_str]
    X_test_str, y_test_str = X[test_idx_str], y_stress[test_idx_str]
    feat_train_str, feat_val_str, feat_test_str = features[train_idx_str], features[val_idx_str], features[test_idx_str]
    groups_train_str = subjects[train_idx_str]
    ds_test_str_eval = datasets[test_idx_str]

    X_train_full, X_val_full, X_test_full = X[train_mask], X[val_mask], X[test_mask]
    y_dep_train_full, y_dep_val_full, y_dep_test_full = y_depression[train_mask], y_depression[val_mask], y_depression[test_mask]
    y_str_train_full, y_str_val_full, y_str_test_full = y_stress[train_mask], y_stress[val_mask], y_stress[test_mask]
    ds_train_full, ds_val_full, ds_test_full = datasets[train_mask], datasets[val_mask], datasets[test_mask]

    print(f"Depression task - Train: {X_train_dep.shape[0]}, Val: {X_val_dep.shape[0]}, Test: {X_test_dep.shape[0]} segments")
    print(f"Stress task     - Train: {X_train_str.shape[0]}, Val: {X_val_str.shape[0]}, Test: {X_test_str.shape[0]} segments")
    print(f"Multi-Task      - Train: {X_train_full.shape[0]}, Val: {X_val_full.shape[0]}, Test: {X_test_full.shape[0]} segments")

    # Add channel dimension to raw EEG for DL models
    X_train_dep_dl = X_train_dep[..., np.newaxis]
    X_val_dep_dl = X_val_dep[..., np.newaxis]
    X_test_dep_dl = X_test_dep[..., np.newaxis]
    X_train_str_dl = X_train_str[..., np.newaxis]
    X_val_str_dl = X_val_str[..., np.newaxis]
    X_test_str_dl = X_test_str[..., np.newaxis]
    X_train_full_dl = X_train_full[..., np.newaxis]
    X_val_full_dl = X_val_full[..., np.newaxis]
    X_test_full_dl = X_test_full[..., np.newaxis]

    # Compute Class Weights to address severe class imbalance
    print("\nComputing class weights for training...")
    # Depression
    classes_dep = np.unique(y_train_dep)
    weights_dep = compute_class_weight('balanced', classes=classes_dep, y=y_train_dep)
    class_weight_dep_dict = {int(c): float(w) for c, w in zip(classes_dep, weights_dep)}
    # Add dummy key -1 for masking support in multi-task loss
    class_weight_dep_dict[-1] = 0.0
    print(f"  Depression Class Weights: {class_weight_dep_dict}")
    
    # Stress
    classes_str = np.unique(y_train_str)
    weights_str = compute_class_weight('balanced', classes=classes_str, y=y_train_str)
    class_weight_str_dict = {int(c): float(w) for c, w in zip(classes_str, weights_str)}
    # Add dummy key -1 for masking support in multi-task loss
    class_weight_str_dict[-1] = 0.0
    print(f"  Stress Class Weights: {class_weight_str_dict}")

    # Compute sample weights for Multi-Task model (Keras 3 does not support class_weight for multi-output models)
    print("Computing sample weights for Multi-Task training...")
    # Depression task
    sample_weight_dep = np.zeros_like(y_dep_train_full, dtype=np.float32)
    valid_dep_mask = (y_dep_train_full != -1)
    sample_weight_dep[valid_dep_mask] = np.where(y_dep_train_full[valid_dep_mask] == 1, class_weight_dep_dict[1], class_weight_dep_dict[0])
    
    # Stress task
    sample_weight_str = np.zeros_like(y_str_train_full, dtype=np.float32)
    valid_str_mask = (y_str_train_full != -1)
    sample_weight_str[valid_str_mask] = np.where(y_str_train_full[valid_str_mask] == 1, class_weight_str_dict[1], class_weight_str_dict[0])

    # Store results
    results = {
        "depression": {},
        "stress": {},
        "multitask": {},
        "dann": {}
    }
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # ==========================================
    # PART 1: DEPRESSION DETECTION MODELS
    # ==========================================
    print("\n--- Training Depression Detection Models ---")
    
    # 1.1 Classical SVM
    print("Fitting SVM on Depression Task (with grid search)...")
    t0 = time.time()
    clf = fit_and_tune_model("SVM", feat_train_dep, y_train_dep, get_svm_pipeline, groups=groups_train_dep)
    results["depression"]["SVM"] = score_with_tuned_threshold(
        y_val_dep, clf.predict_proba(feat_val_dep)[:, 1],
        y_test_dep, clf.predict_proba(feat_test_dep)[:, 1], time.time() - t0,
        subj_test=subj_test_dep)

    # 1.2 Classical Random Forest
    print("Fitting Random Forest on Depression Task (with grid search)...")
    t0 = time.time()
    clf = fit_and_tune_model("Random Forest", feat_train_dep, y_train_dep, get_random_forest_pipeline, groups=groups_train_dep)
    results["depression"]["Random Forest"] = score_with_tuned_threshold(
        y_val_dep, clf.predict_proba(feat_val_dep)[:, 1],
        y_test_dep, clf.predict_proba(feat_test_dep)[:, 1], time.time() - t0,
        subj_test=subj_test_dep)

    # 1.3 Classical XGBoost
    print("Fitting XGBoost on Depression Task (with grid search)...")
    t0 = time.time()
    clf = fit_and_tune_model("XGBoost", feat_train_dep, y_train_dep, get_xgboost_pipeline, groups=groups_train_dep)
    results["depression"]["XGBoost"] = score_with_tuned_threshold(
        y_val_dep, clf.predict_proba(feat_val_dep)[:, 1],
        y_test_dep, clf.predict_proba(feat_test_dep)[:, 1], time.time() - t0,
        subj_test=subj_test_dep)

    # 1.4 DL Models on Depression
    dl_builders_dep = [
        ("1D CNN", lambda: build_1d_cnn(nb_classes=1)),
        ("LSTM", lambda: build_lstm(nb_classes=1)),
        ("CNN-LSTM", lambda: build_cnn_lstm(nb_classes=1)),
        ("EEGNet", lambda: EEGNet(nb_classes=1)),
        ("TSception", lambda: TSception(nb_classes=1)),
        ("EEGTransformer", lambda: EEGTransformer(nb_classes=1))
    ]
    
    for name, build_func in dl_builders_dep:
        print(f"Training {name} on Depression Task...")
        tf.keras.backend.clear_session()
        gc.collect()
        
        t0 = time.time()
        model = build_func()
        # Recompile with custom learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.fit(
            X_train_dep_dl, y_train_dep,
            validation_data=(X_val_dep_dl, y_val_dep),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dep_dict,
            callbacks=[early_stop],
            verbose=0
        )
        results["depression"][name] = score_with_tuned_threshold(
            y_val_dep, model.predict(X_val_dep_dl, verbose=0).flatten(),
            y_test_dep, model.predict(X_test_dep_dl, verbose=0).flatten(), time.time() - t0,
            subj_test=subj_test_dep)

    # ==========================================
    # PART 2: STRESS DETECTION MODELS
    # ==========================================
    print("\n--- Training Stress Detection Models ---")
    
    # 2.1 Classical SVM
    print("Fitting SVM on Stress Task (with grid search & 10k subsampling)...")
    t0 = time.time()
    clf = fit_and_tune_model("SVM", feat_train_str, y_train_str, get_svm_pipeline,
                             groups=groups_train_str, subsample_limit=10000)
    results["stress"]["SVM"] = score_with_tuned_threshold(
        y_val_str, clf.predict_proba(feat_val_str)[:, 1],
        y_test_str, clf.predict_proba(feat_test_str)[:, 1], time.time() - t0,
        ds_test=ds_test_str_eval)

    # 2.2 Classical Random Forest
    print("Fitting Random Forest on Stress Task (with grid search)...")
    t0 = time.time()
    clf = fit_and_tune_model("Random Forest", feat_train_str, y_train_str, get_random_forest_pipeline, groups=groups_train_str)
    results["stress"]["Random Forest"] = score_with_tuned_threshold(
        y_val_str, clf.predict_proba(feat_val_str)[:, 1],
        y_test_str, clf.predict_proba(feat_test_str)[:, 1], time.time() - t0,
        ds_test=ds_test_str_eval)

    # 2.3 Classical XGBoost
    print("Fitting XGBoost on Stress Task (with grid search)...")
    t0 = time.time()
    clf = fit_and_tune_model("XGBoost", feat_train_str, y_train_str, get_xgboost_pipeline, groups=groups_train_str)
    results["stress"]["XGBoost"] = score_with_tuned_threshold(
        y_val_str, clf.predict_proba(feat_val_str)[:, 1],
        y_test_str, clf.predict_proba(feat_test_str)[:, 1], time.time() - t0,
        ds_test=ds_test_str_eval)

    # 2.4 DL Models on Stress
    dl_builders_str = [
        ("1D CNN", lambda: build_1d_cnn(nb_classes=1)),
        ("LSTM", lambda: build_lstm(nb_classes=1)),
        ("CNN-LSTM", lambda: build_cnn_lstm(nb_classes=1)),
        ("EEGNet", lambda: EEGNet(nb_classes=1)),
        ("TSception", lambda: TSception(nb_classes=1)),
        ("EEGTransformer", lambda: EEGTransformer(nb_classes=1))
    ]
    
    for name, build_func in dl_builders_str:
        print(f"Training {name} on Stress Task...")
        tf.keras.backend.clear_session()
        gc.collect()
        
        t0 = time.time()
        model = build_func()
        # Recompile with custom learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.fit(
            X_train_str_dl, y_train_str,
            validation_data=(X_val_str_dl, y_val_str),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_str_dict,
            callbacks=[early_stop],
            verbose=0
        )
        results["stress"][name] = score_with_tuned_threshold(
            y_val_str, model.predict(X_val_str_dl, verbose=0).flatten(),
            y_test_str, model.predict(X_test_str_dl, verbose=0).flatten(), time.time() - t0,
            ds_test=ds_test_str_eval)

    # ==========================================
    # PART 3: ADVANCED MODELS (MULTI-TASK & DANN)
    # ==========================================
    print("\n--- Training Multi-Task and DANN Models ---")
    
    # 3.1 Multi-Task model (using TSception backbone)
    print("Training Multi-Task model (TSception backbone)...")
    tf.keras.backend.clear_session()
    gc.collect()
    
    t0 = time.time()
    mt_model = build_multitask_eeg_model(backbone_type='TSception', Chans=6, Samples=512)
    # Recompile with custom learning rate
    mt_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'depression': masked_binary_crossentropy,
            'stress': masked_binary_crossentropy
        },
        metrics={
            'depression': 'accuracy',
            'stress': 'accuracy'
        }
    )
    
    # Train on full combined dataset using masked crossentropy and sample weights (Keras 3 compatible positional list format)
    mt_model.fit(
        X_train_full_dl,
        [y_dep_train_full, y_str_train_full],
        validation_data=(X_val_full_dl, [y_dep_val_full, y_str_val_full]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=[sample_weight_dep, sample_weight_str],
        callbacks=[early_stop],
        verbose=0
    )
    mt_time = time.time() - t0

    # Evaluate Multi-Task on Depression Task
    results["multitask"]["Depression Branch"] = score_with_tuned_threshold(
        y_val_dep, mt_model.predict(X_val_dep_dl, verbose=0)[0].flatten(),
        y_test_dep, mt_model.predict(X_test_dep_dl, verbose=0)[0].flatten(), mt_time,
        subj_test=subj_test_dep)

    # Evaluate Multi-Task on Stress Task
    results["multitask"]["Stress Branch"] = score_with_tuned_threshold(
        y_val_str, mt_model.predict(X_val_str_dl, verbose=0)[1].flatten(),
        y_test_str, mt_model.predict(X_test_str_dl, verbose=0)[1].flatten(), mt_time,
        ds_test=ds_test_str_eval)

    # 3.2 Domain Adversarial Neural Network (DANN) (for Stress detection across datasets)
    print("Training Domain Adversarial Model (DANN) on Stress Task...")
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Filters for stress task specifically
    ds_train_str = datasets[train_idx_str]
    ds_val_str = datasets[val_idx_str]
    ds_test_str = datasets[test_idx_str]

    # Map dataset labels for stress datasets (SAM40: 1, CogLoad: 2, AMIGOS: 3) to 0, 1, 2 domain tags
    ds_train_str_mapped = np.where(ds_train_str == 1, 0, np.where(ds_train_str == 2, 1, 2))
    ds_val_str_mapped = np.where(ds_val_str == 1, 0, np.where(ds_val_str == 2, 1, 2))
    ds_test_str_mapped = np.where(ds_test_str == 1, 0, np.where(ds_test_str == 2, 1, 2))
    
    # Compute Domain Class Weights to balance domain updates
    unique_domains = np.unique(ds_train_str_mapped)
    weights_dom = compute_class_weight('balanced', classes=unique_domains, y=ds_train_str_mapped)
    class_weight_dom_dict = {int(d): float(w) for d, w in zip(unique_domains, weights_dom)}
    print(f"  Domain Class Weights for GRL: {class_weight_dom_dict}")
    
    t0 = time.time()
    # DANN built with 3 domains (SAM40, CogLoad, AMIGOS) and lower alpha for training stability
    dann_model = build_dann_eeg_model(backbone_type='TSception', nb_classes=1, nb_domains=3, Chans=6, Samples=512, alpha=0.5)
    # Recompile with custom learning rate and softer domain regularization loss weight
    dann_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'task': 'binary_crossentropy',
            'domain': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'task': 1.0,
            'domain': 0.25  # Reduced weight to prevent domain classification from dominating target task learning
        },
        metrics={
            'task': 'accuracy',
            'domain': 'accuracy'
        }
    )
    # Compute DANN sample weights
    sample_weight_task = np.where(y_train_str == 1, class_weight_str_dict[1], class_weight_str_dict[0])
    sample_weight_domain = np.zeros_like(ds_train_str_mapped, dtype=np.float32)
    for d_idx, w_dom in zip(unique_domains, weights_dom):
        sample_weight_domain[ds_train_str_mapped == d_idx] = w_dom

    dann_model.fit(
        X_train_str_dl,
        [y_train_str, ds_train_str_mapped],
        validation_data=(X_val_str_dl, [y_val_str, ds_val_str_mapped]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=[sample_weight_task, sample_weight_domain],
        callbacks=[early_stop],
        verbose=0
    )
    dann_time = time.time() - t0

    # Evaluate DANN task performance on stress test set
    results["dann"]["DANN Stress Task"] = score_with_tuned_threshold(
        y_val_str, dann_model.predict(X_val_str_dl, verbose=0)[0].flatten(),
        y_test_str, dann_model.predict(X_test_str_dl, verbose=0)[0].flatten(), dann_time,
        ds_test=ds_test_str_eval)

    # 4. Write Report
    print(f"\nCompiling metrics and writing report to {REPORT_PATH}...")
    split_info = {
        "dep_train": X_train_dep.shape[0],
        "dep_val": X_val_dep.shape[0],
        "dep_test": X_test_dep.shape[0],
        "str_train": X_train_str.shape[0],
        "str_val": X_val_str.shape[0],
        "str_test": X_test_str.shape[0],
        "full_train": X_train_full.shape[0],
        "full_val": X_val_full.shape[0],
        "full_test": X_test_full.shape[0]
    }
    write_markdown_report(results, split_info)
    print("=== PIPELINE RUN COMPLETE AND REPORT GENERATED ===")

def _metric_cells(metrics):
    """Shared metric columns for every report table."""
    flag = " ⚠️" if metrics['Accuracy'] < metrics['Majority Baseline'] else ""
    return (f"{metrics['Accuracy']:.4f}{flag} | {metrics['Balanced Acc']:.4f} | "
            f"{metrics['Majority Baseline']:.4f} | {metrics['Precision']:.4f} | "
            f"{metrics['Recall']:.4f} | {metrics['F1-Score']:.4f} | "
            f"{metrics['ROC-AUC']:.4f} | {metrics['Threshold']:.2f}")

METRIC_HEADER = ("Accuracy | Balanced Acc | Majority Baseline | Precision | "
                 "Recall | F1-Score | ROC-AUC | Threshold")
METRIC_SEP = ":--- | :--- | :--- | :--- | :--- | :--- | :--- | :---"

def write_markdown_report(results, split_info):
    report_content = f"""# EEG Model Training Metrics Report

This report summarizes the performance metrics of all implemented models on the standardized **6-channel frontal EEG dataset** (Fp1, Fp2, F7, F3, F4, F8) combined from **MODMA, SAM40, Cognitive Load, and AMIGOS**.

Evaluation uses a rigorous **Group-based Train/Val/Test split (70/10/20)** on subject IDs, guaranteeing zero subject overlap across all three splits. The validation split is drawn per-dataset so that early stopping is not steered by a single corpus. All models are trained with **outlier-clipped global segment normalization** (preserving spatial amplitude ratios across channels); deep learning models use **GPU acceleration** with a tuned learning rate, and classical models use **subject-grouped** GridSearch hyperparameter tuning.

> **Reading these numbers:** the **Majority Baseline** column is the accuracy of always predicting the most frequent class. Any model whose Accuracy falls below it (flagged ⚠️) is worse than a constant predictor, and **Balanced Acc** is the meaningful figure there. **Threshold** is the decision cutoff selected on the validation split rather than a fixed 0.5.

---

## 📊 Summary of Combined Dataset Splits
*   **Depression Classification (MODMA)**:
    *   Train / Val / Test: {split_info["dep_train"]} / {split_info["dep_val"]} / {split_info["dep_test"]} segments
*   **Stress Classification (SAM40, Cognitive Load, AMIGOS)**:
    *   Train / Val / Test: {split_info["str_train"]} / {split_info["str_val"]} / {split_info["str_test"]} segments
*   **Multi-Task (Full Dataset)**:
    *   Train / Val / Test: {split_info["full_train"]} / {split_info["full_val"]} / {split_info["full_test"]} segments

---

## 🧠 Part 1: Depression Detection Results (Clinical Classification)

The models below were trained to classify Major Depressive Disorder (MDD) vs. Healthy Controls (HC) using the MODMA dataset.

| Model Class | Model | """ + METRIC_HEADER + """ | Training Time |
| :--- | :--- | """ + METRIC_SEP + """ | :--- |
"""
    # Append Depression results
    for model_name, metrics in results["depression"].items():
        m_class = "Classical ML (Feature-based)" if model_name in ["SVM", "Random Forest", "XGBoost"] else "Deep Learning (Raw EEG)"
        report_content += f"| {m_class} | {model_name} | {_metric_cells(metrics)} | {metrics['Time']:.2f}s |\n"

    report_content += """
---

## ⚡ Part 2: Stress Detection Results (State Classification)

The models below were trained to classify cognitive stress/arousal states vs. relaxed baselines using SAM40, Cognitive Load, and AMIGOS datasets.

| Model Class | Model | """ + METRIC_HEADER + """ | Training Time |
| :--- | :--- | """ + METRIC_SEP + """ | :--- |
"""
    # Append Stress results
    for model_name, metrics in results["stress"].items():
        m_class = "Classical ML (Feature-based)" if model_name in ["SVM", "Random Forest", "XGBoost"] else "Deep Learning (Raw EEG)"
        report_content += f"| {m_class} | {model_name} | {_metric_cells(metrics)} | {metrics['Time']:.2f}s |\n"

    # Per-corpus stress breakdown. The pooled row above is ~78% AMIGOS, whose label
    # is not decodable, so it understates performance on the task-based corpora.
    per_ds = {n: m["PerDataset"] for n, m in results["stress"].items() if m.get("PerDataset")}
    if per_ds:
        corpora = sorted({c for v in per_ds.values() for c in v})
        report_content += """
### Stress ROC-AUC by source corpus

> The pooled row above is dominated (~78%) by AMIGOS, whose self-reported arousal label
> is **not EEG-decodable** from this montage (Phase 1: 0.4875 within-subject AUC, and a
> per-subject median split changes it by only −0.004). Read this breakdown, not the
> pooled figure. Stress should be reported on **SAM40 / CogLoad** only.

| Model | """ + " | ".join(corpora) + """ |
| :--- | """ + " | ".join([":---"] * len(corpora)) + """ |
"""
        for name, vals in per_ds.items():
            cells = " | ".join(f"{vals[c]:.4f}" if c in vals else "—" for c in corpora)
            report_content += f"| {name} | {cells} |\n"

    # Subject-level depression. Every segment of a subject shares one label, so the
    # subject is the meaningful unit; segment metrics on overlapping windows inflate.
    subj = {n: m["SubjectLevel"] for n, m in results["depression"].items() if m.get("SubjectLevel")}
    if subj:
        n_sub = next(iter(subj.values()))["n_subjects"]
        report_content += f"""
### Depression at the subject level ({n_sub} held-out subjects)

> Every segment from a subject carries the same label, so the clinically meaningful
> question is whether the **subject** is MDD. Note this is still a single split — see
> `Results/phase1/depression_cv.json` for out-of-fold estimates over all 53 subjects,
> which are the reportable numbers. Single-split figures here inflate deep-model AUC
> by roughly 0.21.

| Model | Subject ROC-AUC | Subject Balanced Acc |
| :--- | :--- | :--- |
"""
        for name, m in subj.items():
            report_content += f"| {name} | {m['auc']:.4f} | {m['balanced_acc']:.4f} |\n"

    report_content += """
---

## 🧬 Part 3: Advanced Models (Multi-Task & Domain Adversarial)

### 3.1 Shared-Backbone Multi-Task Network
A single **TSception** model backbone trained on the entire combined dataset to simultaneously output depression and stress predictions using a custom masked cross-entropy loss function.

| Multi-Task Evaluation Branch | """ + METRIC_HEADER + """ | Joint Training Time |
| :--- | """ + METRIC_SEP + """ | :--- |
"""
    # Append Multi-task results
    mt_time = results["multitask"]["Depression Branch"]["Time"]
    for branch_name, metrics in results["multitask"].items():
        report_content += f"| {branch_name} | {_metric_cells(metrics)} | {mt_time:.2f}s |\n"

    report_content += """
### 3.2 Domain Adversarial Neural Network (DANN)
A **TSception** model incorporating a custom **Gradient Reversal Layer (GRL)**. It is trained to perform stress detection ('task' branch) while learning domain-invariant features by trying to confuse a dataset-origin classifier ('domain' branch).

| Model Configuration | """ + METRIC_HEADER + """ | Training Time |
| :--- | """ + METRIC_SEP + """ | :--- |
"""
    # Append DANN results
    for model_name, metrics in results["dann"].items():
        report_content += f"| {model_name} | {_metric_cells(metrics)} | {metrics['Time']:.2f}s |\n"

    report_content += """
---

## 📈 Key Findings and Recommendations

0. **Label corrections applied to the combined dataset**:
   * **AMIGOS** was reading `labels_selfassessment` index 1 (**valence**) while treating it as arousal. Per `SelfAsessment.xlsx` the block order is `arousal, valence, dominance, ...`, so index 0 is arousal; this was verified against the spreadsheet for 80 subject/trial pairs. Since arousal and valence correlate at only −0.11 in this data, ~71% of the corpus had been labelled by a near-orthogonal variable.
   * **Cognitive Load** hardcoded every file to stress-positive, including its 30 `natural-*` rest recordings. Those now form the dataset's baseline class, which previously did not exist.

1. **Impact of Global Segment Normalization & Artifact Clipping**:
   * Outlier-clipped global segment-level normalization preserves spatial amplitude ratios across the channels, which are essential for frontal asymmetry and spatial filtering.
   * Expanding features with relative power, spectral complexity, and amplitude characteristics improves baseline classical machine learning models.

2. **Hyperparameter Grid Searching**:
   * Tuning classifiers using cross-validated GridSearchCV isolates optimal hyperparameters (e.g. C limits in SVM, max depths in Random Forest, learning rates in XGBoost) preventing underfitting and severe overfitting.

3. **Optimized Deep Learning Convergence**:
   * Operating at a lower, tuned learning rate (5e-4) with early stopping patience extended to 5 epochs stabilizes SGD optimization paths, allowing CNNs, LSTMs, and Transformers to identify robust local minima from raw EEG signals.
"""
    
    with open(REPORT_PATH, "w") as f:
        f.write(report_content)

if __name__ == "__main__":
    main()
