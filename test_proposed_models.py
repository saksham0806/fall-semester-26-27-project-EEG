import os
import numpy as np
import tensorflow as tf

# Set logging level for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import package models
from proposed_model import (
    extract_psd_features,
    extract_differential_entropy,
    extract_faa_features,
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

def run_tests():
    print("=== Starting Proposed Model Verification Tests ===")
    
    # 1. Generate Dummy Data (32 segments, 6 channels, 512 time steps)
    print("\n[1/6] Generating dummy EEG segment data...")
    X_raw = np.random.normal(size=(32, 6, 512))
    # Normalize segment-wise per channel (Z-score)
    mean = np.mean(X_raw, axis=-1, keepdims=True)
    std = np.std(X_raw, axis=-1, keepdims=True) + 1e-8
    X = (X_raw - mean) / std
    
    print(f"  EEG Data shape: {X.shape} (Expected: (32, 6, 512))")
    assert X.shape == (32, 6, 512)
    print("  [OK] Data generation and normalization complete.")

    # 2. Test Feature Extraction
    print("\n[2/6] Testing feature extraction modules...")
    psd = extract_psd_features(X)
    de = extract_differential_entropy(X)
    faa = extract_faa_features(X)
    all_feats = extract_all_features(X)
    
    print(f"  PSD shape: {psd.shape} (Expected: (32, 30))")
    print(f"  DE shape: {de.shape} (Expected: (32, 30))")
    print(f"  FAA shape: {faa.shape} (Expected: (32, 3))")
    print(f"  All concatenated features shape: {all_feats.shape} (Expected: (32, 63))")
    
    assert psd.shape == (32, 30)
    assert de.shape == (32, 30)
    assert faa.shape == (32, 3)
    assert all_feats.shape == (32, 63)
    print("  [OK] All feature extraction modules passed successfully.")

    # 3. Test Classical ML Pipelines
    print("\n[3/6] Testing classical ML pipelines...")
    y_class = np.random.choice([0, 1], size=32)
    
    for name, get_pipeline in [
        ("SVM", get_svm_pipeline),
        ("Random Forest", get_random_forest_pipeline),
        ("XGBoost/GBDT", get_xgboost_pipeline)
    ]:
        pipeline = get_pipeline()
        pipeline.fit(all_feats, y_class)
        preds = pipeline.predict(all_feats)
        probs = pipeline.predict_proba(all_feats)
        print(f"  {name} - Preds shape: {preds.shape}, Probs shape: {probs.shape}")
        assert preds.shape == (32,)
        assert probs.shape == (32, 2)
    print("  [OK] All classical ML pipelines trained and predicted successfully.")

    # Prepare inputs for Deep Learning (Requires channel dim: (32, 6, 512, 1))
    X_dl = X[..., np.newaxis]
    y_dl = np.random.choice([0, 1], size=32)
    
    # 4. Test Standard Deep Learning Baselines
    print("\n[4/6] Testing standard deep learning baselines...")
    for name, build_func in [
        ("1D CNN", build_1d_cnn),
        ("LSTM", build_lstm),
        ("CNN-LSTM", build_cnn_lstm)
    ]:
        model = build_func(nb_classes=1, Chans=6, Samples=512)
        preds = model.predict(X_dl, verbose=0)
        print(f"  {name} - Prediction shape: {preds.shape} (Expected: (32, 1))")
        assert preds.shape == (32, 1)
    print("  [OK] All standard DL baselines compiled and forward-passed successfully.")

    # 5. Test EEG-Specific Architectures
    print("\n[5/6] Testing EEG-specific models...")
    for name, build_func in [
        ("EEGNet", lambda: EEGNet(nb_classes=1, Chans=6, Samples=512)),
        ("TSception", lambda: TSception(nb_classes=1, Chans=6, Samples=512)),
        ("EEGTransformer", lambda: EEGTransformer(nb_classes=1, Chans=6, Samples=512))
    ]:
        model = build_func()
        preds = model.predict(X_dl, verbose=0)
        print(f"  {name} - Prediction shape: {preds.shape} (Expected: (32, 1))")
        assert preds.shape == (32, 1)
    print("  [OK] All EEG-specific models compiled and forward-passed successfully.")

    # 6. Test Advanced Multi-Task and DANN Models
    print("\n[6/6] Testing multi-task & DANN models and custom loss...")
    
    # Test masked loss directly
    y_true_dummy = tf.constant([[1.0], [0.0], [-1.0], [1.0]])
    y_pred_dummy = tf.constant([[0.9], [0.1], [0.5], [0.2]])
    loss_val = masked_binary_crossentropy(y_true_dummy, y_pred_dummy)
    print(f"  Custom Masked loss for dummy inputs: {loss_val.numpy():.4f}")
    assert not np.isnan(loss_val.numpy())
    
    # Test Multi-task
    mt_model = build_multitask_eeg_model(backbone_type='TSception', Chans=6, Samples=512)
    preds = mt_model.predict(X_dl, verbose=0)
    print(f"  Multi-Task - Outputs: {len(preds)}, Depression shape: {preds[0].shape}, Stress shape: {preds[1].shape}")
    assert len(preds) == 2
    assert preds[0].shape == (32, 1)
    assert preds[1].shape == (32, 1)
    
    # Test DANN
    dann_model = build_dann_eeg_model(backbone_type='TSception', nb_classes=1, nb_domains=4, Chans=6, Samples=512)
    preds = dann_model.predict(X_dl, verbose=0)
    print(f"  DANN - Outputs: {len(preds)}, Task shape: {preds[0].shape}, Domain shape: {preds[1].shape}")
    assert len(preds) == 2
    assert preds[0].shape == (32, 1)
    assert preds[1].shape == (32, 4)
    
    print("  [OK] Multi-task and DANN models compiled and forward-passed successfully.")
    
    print("\n=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
