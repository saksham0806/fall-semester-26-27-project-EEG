import os
import scipy.io
import mne
import numpy as np

REST_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA/EEG_128channels_resting_lanzhou_2015"
ERP_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA/EEG_128channels_ERP_lanzhou_2015"

def inspect_resting():
    print("\n--- Inspecting Resting State Data (.mat) ---")
    files = [f for f in os.listdir(REST_DIR) if f.endswith('.mat') and 'chan_info' not in f]
    if not files:
        print("No .mat files found.")
        return

    file_path = os.path.join(REST_DIR, files[0])
    print(f"Loading {file_path}...")
    try:
        mat = scipy.io.loadmat(file_path)
        print("Keys:", mat.keys())
        for key in mat.keys():
            if not key.startswith('__'):
                data = mat[key]
                print(f"Key: {key}, Type: {type(data)}")
                if isinstance(data, np.ndarray):
                    print(f"Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading .mat: {e}")

def inspect_erp():
    print("\n--- Inspecting ERP Data (.raw) ---")
    files = [f for f in os.listdir(ERP_DIR) if f.endswith('.raw')]
    if not files:
        print("No .raw files found.")
        return

    file_path = os.path.join(ERP_DIR, files[0])
    print(f"Loading {file_path}...")
    try:
        # Try EGI first as .raw is common for EGI
        raw = mne.io.read_raw_egi(file_path, preload=False, verbose=False)
        print("Loaded with MNE (EGI).")
        print(raw.info)
        print(f"Data shape: {raw.n_times} samples x {len(raw.ch_names)} channels")
    except Exception as e:
        print(f"Error loading with MNE: {e}")
        # Try Binary?

if __name__ == "__main__":
    inspect_resting()
    inspect_erp()
