import os
import json
import numpy as np

# Paths resolved dynamically relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")

def verify():
    print("=== Combined EEG Dataset Verification ===")
    
    # Files to verify
    files = ["X.npy", "y_depression.npy", "y_stress.npy", "subjects.npy", "datasets.npy", "metadata.json"]
    
    # Check file existence
    print("Checking file existence...")
    for f in files:
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(path):
            print(f"  [OK] Found {f} ({os.path.getsize(path) / (1024*1024):.2f} MB)")
        else:
            print(f"  [ERROR] Missing {f}!")
            return
            
    # Load metadata
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)
        
    print("\nMetadata summary:")
    print(json.dumps(meta, indent=4))
    
    # Load arrays
    print("\nLoading arrays...")
    X = np.load(os.path.join(OUTPUT_DIR, "X.npy"))
    y_depression = np.load(os.path.join(OUTPUT_DIR, "y_depression.npy"))
    y_stress = np.load(os.path.join(OUTPUT_DIR, "y_stress.npy"))
    subjects = np.load(os.path.join(OUTPUT_DIR, "subjects.npy"), allow_pickle=True)
    datasets = np.load(os.path.join(OUTPUT_DIR, "datasets.npy"))
    
    print("\nArray details:")
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y_depression shape: {y_depression.shape}, dtype: {y_depression.dtype}")
    print(f"  y_stress shape: {y_stress.shape}, dtype: {y_stress.dtype}")
    print(f"  subjects shape: {subjects.shape}, dtype: {subjects.dtype}")
    print(f"  datasets shape: {datasets.shape}, dtype: {datasets.dtype}")
    
    # Shape checks
    assert X.shape[0] == y_depression.shape[0] == y_stress.shape[0] == subjects.shape[0] == datasets.shape[0], "Length mismatch between arrays!"
    assert X.shape[1] == 6, "Channel count should be 6!"
    assert X.shape[2] == 512, "Window size should be 512!"
    print("  [OK] Shape alignments are correct.")
    
    # Normalization check
    print("\nVerifying Z-score normalization...")
    # Compute mean and std for a sample segment globally (since global normalization is used to preserve spatial amplitude ratio across channels)
    sample_idx = 42
    sample_mean_global = np.mean(X[sample_idx])
    sample_std_global = np.std(X[sample_idx])
    print(f"  Sample segment #{sample_idx} global mean: {sample_mean_global}")
    print(f"  Sample segment #{sample_idx} global std: {sample_std_global}")
    
    # Verify they are very close to 0 and 1
    assert np.allclose(sample_mean_global, 0.0, atol=1e-5), "Z-score normalization mean check failed!"
    assert np.allclose(sample_std_global, 1.0, atol=1e-5), "Z-score normalization std check failed!"
    print("  [OK] Z-score normalization verified successfully.")
    
    # Label mapping sanity checks
    print("\nVerifying label mapping sanity...")
    
    # MODMA (0)
    modma_mask = (datasets == 0)
    modma_dep = y_depression[modma_mask]
    modma_str = y_stress[modma_mask]
    print(f"  MODMA (0):")
    print(f"    y_depression unique values: {np.unique(modma_dep)} (expected [0 1])")
    print(f"    y_stress unique values: {np.unique(modma_str)} (expected [-1])")
    assert set(np.unique(modma_dep)) == {0, 1}, "MODMA depression labels invalid!"
    assert np.unique(modma_str).tolist() == [-1], "MODMA stress labels invalid!"
    
    # SAM40 (1)
    sam_mask = (datasets == 1)
    sam_dep = y_depression[sam_mask]
    sam_str = y_stress[sam_mask]
    print(f"  SAM40 (1):")
    print(f"    y_depression unique values: {np.unique(sam_dep)} (expected [-1])")
    print(f"    y_stress unique values: {np.unique(sam_str)} (expected [0 1])")
    assert np.unique(sam_dep).tolist() == [-1], "SAM40 depression labels invalid!"
    assert set(np.unique(sam_str)) == {0, 1}, "SAM40 stress labels invalid!"
    
    # Cognitive Load (2)
    cog_mask = (datasets == 2)
    cog_dep = y_depression[cog_mask]
    cog_str = y_stress[cog_mask]
    print(f"  Cognitive Load (2):")
    print(f"    y_depression unique values: {np.unique(cog_dep)} (expected [-1])")
    print(f"    y_stress unique values: {np.unique(cog_str)} (expected [0 1])")
    assert np.unique(cog_dep).tolist() == [-1], "Cognitive Load depression labels invalid!"
    assert set(np.unique(cog_str)) == {0, 1}, "Cognitive Load stress labels invalid!"
    
    # AMIGOS (3)
    amigos_mask = (datasets == 3)
    amigos_dep = y_depression[amigos_mask]
    amigos_str = y_stress[amigos_mask]
    print(f"  AMIGOS (3):")
    print(f"    y_depression unique values: {np.unique(amigos_dep)} (expected [-1])")
    print(f"    y_stress unique values: {np.unique(amigos_str)} (expected [0 1])")
    assert np.unique(amigos_dep).tolist() == [-1], "AMIGOS depression labels invalid!"
    assert set(np.unique(amigos_str)) == {0, 1}, "AMIGOS stress labels invalid!"
    
    print("\n  [OK] Label mappings are correctly isolated per dataset.")
    print("\n=== VERIFICATION SUCCESSFULLY PASSED! ===")

if __name__ == "__main__":
    verify()
