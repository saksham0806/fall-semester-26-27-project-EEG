import os
import glob
import json
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample

# Paths
BASE_DIR = "/run/media/Saksham/Saksham/Code/Projects/fall-semester-26-27-project-EEG/datasets"
MODMA_DIR = os.path.join(BASE_DIR, "MODMA/EEG_128channels_resting_lanzhou_2015")
SAM40_DIR = os.path.join(BASE_DIR, "SAM40")
COG_DIR = os.path.join(BASE_DIR, "Cognitive Load Assessment Through EEG A Dataset from Arithmetic and Stroop Tasks")
AMIGOS_DIR = os.path.join(BASE_DIR, "AMIGOS DATASET")
OUTPUT_DIR = "/run/media/Saksham/Saksham/Code/Projects/fall-semester-26-27-project-EEG/datasets/combined_data"

# Standard parameters
TARGET_RATE = 128
WINDOW_SIZE = 512  # 4 seconds at 128 Hz
STRIDE = 256       # 2 seconds stride (50% overlap)
NUM_CHANNELS = 6

def z_score_normalize(segment):
    """
    Z-score normalize the entire segment globally to preserve relative channel scales.
    Uses Median and Median Absolute Deviation (MAD) to robustly clip extreme outliers
    from sensor movement before scaling.
    """
    median = np.median(segment)
    mad = np.median(np.abs(segment - median)) + 1e-8
    # Clip to +/- 5 MADs to suppress motion artifact spikes
    segment_clipped = np.clip(segment, median - 5 * mad, median + 5 * mad)
    
    mean = np.mean(segment_clipped)
    std = np.std(segment_clipped) + 1e-8
    return (segment_clipped - mean) / std

def segment_and_process(data, target_rate, native_rate, subject_id, dataset_id, y_dep, y_str):
    """
    Resample, normalize, segment and label data.
    data shape: (channels, native_samples)
    """
    # 1. Resample to 128 Hz
    if native_rate != target_rate:
        target_len = int(data.shape[1] * target_rate / native_rate)
        data_resampled = resample(data, target_len, axis=1)
    else:
        data_resampled = data.copy()
        
    n_channels, n_samples = data_resampled.shape
    
    segments = []
    labels_dep = []
    labels_str = []
    subjects = []
    datasets = []
    
    # 2. Windowing
    for start in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
        seg = data_resampled[:, start:start+WINDOW_SIZE]
        
        # Normalize
        seg_norm = z_score_normalize(seg)
        
        segments.append(seg_norm)
        labels_dep.append(y_dep)
        labels_str.append(y_str)
        subjects.append(subject_id)
        datasets.append(dataset_id)
        
    return segments, labels_dep, labels_str, subjects, datasets

def load_modma():
    print("\n--- Processing MODMA ---")
    excel_path = os.path.join(MODMA_DIR, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
    if not os.path.exists(excel_path):
        print("MODMA Excel file not found!")
        return [], [], [], [], []
        
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    df['subject id'] = df['subject id'].astype(str)
    meta = df.set_index('subject id')
    
    mat_files = glob.glob(os.path.join(MODMA_DIR, "*.mat"))
    
    segments_all, dep_all, str_all, subj_all, ds_all = [], [], [], [], []
    
    # Channels mapping (0-indexed for E22, E9, E33, E24, E124, E122)
    # Fp1 -> 21, Fp2 -> 8, F7 -> 32, F3 -> 23, F4 -> 123, F8 -> 121
    channels_idx = [21, 8, 32, 23, 123, 121]
    
    for f in mat_files:
        basename = os.path.basename(f)
        # Extract numeric subject ID from filename (e.g. 02010002 -> 2010002)
        candidates = ''.join([c if c.isdigit() else ' ' for c in basename]).split()
        if not candidates: continue
        full_id = candidates[0]
        clean_id = str(int(full_id))
        
        if clean_id not in meta.index: continue
        
        label_str = meta.loc[clean_id, 'type']
        y_dep = 1 if label_str == 'MDD' else 0
        y_str = -1 # Not applicable for resting state
        
        try:
            mat = scipy.io.loadmat(f)
            # Find data key
            data = None
            for k, v in mat.items():
                if isinstance(v, np.ndarray) and (v.shape[0] == 128 or v.shape[0] == 129):
                    data = v[channels_idx, :]
                    break
            if data is None: continue
            
            segs, deps, strs, subjs, dss = segment_and_process(
                data, TARGET_RATE, 250, f"modma_{clean_id}", 0, y_dep, y_str
            )
            segments_all.extend(segs)
            dep_all.extend(deps)
            str_all.extend(strs)
            subj_all.extend(subjs)
            ds_all.extend(dss)
            print(f"Processed MODMA Subject {clean_id}: {len(segs)} segments")
        except Exception as e:
            print(f"Error processing MODMA file {f}: {e}")
            
    return segments_all, dep_all, str_all, subj_all, ds_all

def load_sam40():
    print("\n--- Processing SAM40 ---")
    filtered_dir = os.path.join(SAM40_DIR, "filtered_data")
    mat_files = glob.glob(os.path.join(filtered_dir, "*.mat"))
    
    # Channels mapping (Fp1 -> 2, Fp2 -> 31, F7 -> 3, F3 -> 4, F4 -> 29, F8 -> 30)
    channels_idx = [2, 31, 3, 4, 29, 30]
    
    segments_all, dep_all, str_all, subj_all, ds_all = [], [], [], [], []
    
    for f in mat_files:
        basename = os.path.basename(f)
        # Parse subject ID and task
        parts = basename.replace('.mat', '').split('_')
        if len(parts) < 3: continue
        
        task = parts[0]
        sub_id = None
        for i, p in enumerate(parts):
            if p == 'sub' and i + 1 < len(parts):
                sub_id = parts[i+1]
                break
        if sub_id is None: continue
        
        y_dep = -1 # Not applicable
        if task.lower() == 'relax':
            y_str = 0
        elif task.lower() in ['arithmetic', 'stroop', 'mirror', 'mirror_image']:
            y_str = 1
        else:
            continue
            
        try:
            mat = scipy.io.loadmat(f)
            if 'Clean_data' not in mat: continue
            data = mat['Clean_data'][channels_idx, :]
            
            # Since native is 256 Hz, we can downsample by exactly 2
            data_downsampled = data[:, ::2]
            
            segs, deps, strs, subjs, dss = segment_and_process(
                data_downsampled, TARGET_RATE, 128, f"sam40_{sub_id}", 1, y_dep, y_str
            )
            segments_all.extend(segs)
            dep_all.extend(deps)
            str_all.extend(strs)
            subj_all.extend(subjs)
            ds_all.extend(dss)
        except Exception as e:
            print(f"Error processing SAM40 file {f}: {e}")
            
    print(f"Processed SAM40: {len(segments_all)} segments total")
    return segments_all, dep_all, str_all, subj_all, ds_all

def load_cognitive_load():
    print("\n--- Processing Cognitive Load ---")
    stroop_files = glob.glob(os.path.join(COG_DIR, "raw_data/Stroop_Data/*.txt"))
    arith_files = glob.glob(os.path.join(COG_DIR, "raw_data/Arithmetic_Data*/*.txt"))
    
    # 0-indexed columns to extract: col 1, col 2, col 3, col 4, col 6, col 7
    channels_idx = [1, 2, 3, 4, 6, 7]
    
    segments_all, dep_all, str_all, subj_all, ds_all = [], [], [], [], []
    
    all_files = [(f, 'stroop') for f in stroop_files] + [(f, 'arithmetic') for f in arith_files]
    
    for f, task in all_files:
        basename = os.path.basename(f)
        candidates = ''.join([c if c.isdigit() else ' ' for c in basename]).split()
        sub_id = candidates[0] if candidates else basename
        
        y_dep = -1
        y_str = 1 # Tasks are active cognitive load (Stress/Workload)
        
        try:
            df = pd.read_csv(f, header=None, comment='#', on_bad_lines='skip')
            data = df.iloc[:, channels_idx].values.T # Shape: (6, n_samples)
            
            segs, deps, strs, subjs, dss = segment_and_process(
                data, TARGET_RATE, 250, f"cogload_{sub_id}", 2, y_dep, y_str
            )
            segments_all.extend(segs)
            dep_all.extend(deps)
            str_all.extend(strs)
            subj_all.extend(subjs)
            ds_all.extend(dss)
        except Exception as e:
            print(f"Error processing Cognitive Load file {f}: {e}")
            
    print(f"Processed Cognitive Load: {len(segments_all)} segments total")
    return segments_all, dep_all, str_all, subj_all, ds_all

def load_amigos():
    print("\n--- Processing AMIGOS ---")
    mat_files = glob.glob(os.path.join(AMIGOS_DIR, "Data_Preprocessed_P*/Data_Preprocessed_P*.mat"))
    
    # 0-indexed columns in AMIGOS: 0: AF3, 13: AF4, 1: F7, 2: F3, 11: F4, 12: F8
    channels_idx = [0, 13, 1, 2, 11, 12]
    
    segments_all, dep_all, str_all, subj_all, ds_all = [], [], [], [], []
    
    for f in mat_files:
        basename = os.path.basename(f)
        sub_id = basename.replace('Data_Preprocessed_', '').replace('.mat', '')
        
        try:
            mat = scipy.io.loadmat(f)
            joined_data = mat['joined_data']
            labels_self = mat['labels_selfassessment']
            
            n_trials = joined_data.shape[1]
            for trial_idx in range(n_trials):
                trial_sig = joined_data[0, trial_idx]
                trial_labels = labels_self[0, trial_idx]
                
                if not isinstance(trial_sig, np.ndarray) or trial_sig.size == 0:
                    continue
                if not isinstance(trial_labels, np.ndarray) or trial_labels.size == 0:
                    continue
                    
                data = trial_sig.T[channels_idx, :]
                
                # Arousal is at index 1
                arousal = trial_labels[0, 1]
                y_dep = -1
                y_str = 1 if arousal >= 5.0 else 0
                
                segs, deps, strs, subjs, dss = segment_and_process(
                    data, TARGET_RATE, 128, f"amigos_{sub_id}", 3, y_dep, y_str
                )
                segments_all.extend(segs)
                dep_all.extend(deps)
                str_all.extend(strs)
                subj_all.extend(subjs)
                ds_all.extend(dss)
        except Exception as e:
            print(f"Error processing AMIGOS file {f}: {e}")
            
    print(f"Processed AMIGOS: {len(segments_all)} segments total")
    return segments_all, dep_all, str_all, subj_all, ds_all

def main():
    # Make sure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load each dataset
    m_seg, m_dep, m_str, m_sub, m_ds = load_modma()
    s_seg, s_dep, s_str, s_sub, s_ds = load_sam40()
    c_seg, c_dep, c_str, c_sub, c_ds = load_cognitive_load()
    a_seg, a_dep, a_str, a_sub, a_ds = load_amigos()
    
    # 2. Combine all lists
    print("\n--- Compiling Combined Dataset ---")
    all_segments = m_seg + s_seg + c_seg + a_seg
    all_dep = m_dep + s_dep + c_dep + a_dep
    all_str = m_str + s_str + c_str + a_str
    all_sub = m_sub + s_sub + c_sub + a_sub
    all_ds = m_ds + s_ds + c_ds + a_ds
    
    # Convert to numpy arrays
    print("Converting to NumPy arrays...")
    X = np.array(all_segments, dtype=np.float32)
    y_depression = np.array(all_dep, dtype=np.int8)
    y_stress = np.array(all_str, dtype=np.int8)
    subjects = np.array(all_sub, dtype=object)
    datasets = np.array(all_ds, dtype=np.int8)
    
    print(f"Combined X shape: {X.shape}")
    print(f"Combined y_depression shape: {y_depression.shape}")
    print(f"Combined y_stress shape: {y_stress.shape}")
    print(f"Combined subjects shape: {subjects.shape}")
    print(f"Combined datasets shape: {datasets.shape}")
    
    # 3. Save files
    print("\nSaving datasets to combined_data/ ...")
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_depression.npy"), y_depression)
    np.save(os.path.join(OUTPUT_DIR, "y_stress.npy"), y_stress)
    np.save(os.path.join(OUTPUT_DIR, "subjects.npy"), subjects)
    np.save(os.path.join(OUTPUT_DIR, "datasets.npy"), datasets)
    
    # 4. Generate metadata statistics
    metadata = {
        "summary": {
            "total_segments": len(X),
            "channels": NUM_CHANNELS,
            "window_size_samples": WINDOW_SIZE,
            "sampling_rate_hz": TARGET_RATE
        },
        "datasets": {
            "MODMA (0)": {
                "total_segments": len(m_seg),
                "depression_mdd_count": int(np.sum(y_depression[datasets == 0] == 1)),
                "depression_hc_count": int(np.sum(y_depression[datasets == 0] == 0))
            },
            "SAM40 (1)": {
                "total_segments": len(s_seg),
                "stress_active_count": int(np.sum(y_stress[datasets == 1] == 1)),
                "stress_relax_count": int(np.sum(y_stress[datasets == 1] == 0))
            },
            "Cognitive_Load (2)": {
                "total_segments": len(c_seg),
                "stress_active_count": int(np.sum(y_stress[datasets == 2] == 1)),
                "stress_relax_count": int(np.sum(y_stress[datasets == 2] == 0))
            },
            "AMIGOS (3)": {
                "total_segments": len(a_seg),
                "stress_active_count": int(np.sum(y_stress[datasets == 3] == 1)),
                "stress_relax_count": int(np.sum(y_stress[datasets == 3] == 0))
            }
        }
    }
    
    # Save metadata JSON
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Combined dataset creation complete!")
    print(json.dumps(metadata, indent=4))

if __name__ == "__main__":
    main()
