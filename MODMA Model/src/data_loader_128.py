import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import mne
from scipy.signal import resample

class MODMADataLoader128:
    def __init__(self, rest_dir, erp_dir, excel_rest, excel_erp, target_rate=128):
        self.rest_dir = rest_dir
        self.erp_dir = erp_dir
        self.excel_rest = excel_rest
        self.excel_erp = excel_erp
        self.target_rate = target_rate
        
        self.subjects_df = None
        self.file_map = {'rest': {}, 'erp': {}}
        self.common_subjects = []
        
        self._load_metadata()
        self._map_files()
        self._find_common_subjects()

    def _load_metadata(self):
        # Load Resting Metadata
        df_rest = pd.read_excel(self.excel_rest)
        df_rest.columns = df_rest.columns.str.strip()
        if 'subject id' in df_rest.columns:
            df_rest['subject id'] = df_rest['subject id'].astype(str)
        self.meta_rest = df_rest.set_index('subject id')
        
        # Load ERP Metadata
        df_erp = pd.read_excel(self.excel_erp)
        df_erp.columns = df_erp.columns.str.strip()
        if 'subject id' in df_erp.columns:
            df_erp['subject id'] = df_erp['subject id'].astype(str)
        self.meta_erp = df_erp.set_index('subject id')

    def _map_files(self):
        # Map Resting (.mat)
        rest_files = glob.glob(os.path.join(self.rest_dir, "*.mat"))
        for f in rest_files:
            if 'chan_info' in f: continue
            basename = os.path.basename(f)
            # Extarct ID: 02010002... -> remove leading 0 -> 2010002
            # Heuristic: find first sequence of digits
            try:
                # Based on file inspection, id is usually at start
                # e.g. 02010002rest...
                candidates = ''.join([c if c.isdigit() else ' ' for c in basename]).split()
                if candidates:
                    # The ID usually matches the excel ID which doesn't have the leading 0
                    # Excel: 2010002. File: 02010002.
                    # So we take candidate[0] and strip leading zero if present.
                    # ACTUALLY, checking previous investigation: 
                    # Excel IDs: 2010002. File content IDs: 02010002.
                    # We need to align.
                    # Let's clean the candidate.
                    
                    full_id = candidates[0]
                    clean_id = str(int(full_id)) # This removes leading zeros
                    
                    self.file_map['rest'][clean_id] = f
            except:
                pass

        # Map ERP (.raw)
        erp_files = glob.glob(os.path.join(self.erp_dir, "*.raw"))
        for f in erp_files:
            basename = os.path.basename(f)
            try:
                candidates = ''.join([c if c.isdigit() else ' ' for c in basename]).split()
                if candidates:
                    full_id = candidates[0]
                    clean_id = str(int(full_id))
                    self.file_map['erp'][clean_id] = f
            except:
                pass
                
        print(f"Mapped {len(self.file_map['rest'])} Resting files and {len(self.file_map['erp'])} ERP files.")

    def _find_common_subjects(self):
        rest_subs = set(self.file_map['rest'].keys())
        erp_subs = set(self.file_map['erp'].keys())
        self.common_subjects = sorted(list(rest_subs.intersection(erp_subs)))
        print(f"Found {len(self.common_subjects)} common subjects.")

    def load_subject_data(self, subject_id, modality='rest'):
        """
        Load data for a subject.
        Returns: (n_channels, n_samples) numpy array
        """
        if subject_id not in self.file_map[modality]:
            return None
        
        path = self.file_map[modality][subject_id]
        
        try:
            if modality == 'rest':
                mat = scipy.io.loadmat(path)
                # Find the data key. It's usually the one that is not header/version/globals
                # and has shape (129, N)
                for k, v in mat.items():
                    if isinstance(v, np.ndarray) and v.shape[0] == 129:
                        # 129 channels (128 EEG + 1 Ref/VEO?). We usually take first 128.
                        data = v[:128, :] 
                        # Resample if needed
                        # Original rate is usually 250Hz.
                        if self.target_rate != 250:
                            n_samples = int(data.shape[1] * self.target_rate / 250)
                            data = resample(data, n_samples, axis=1)
                        return data
                return None
                
            elif modality == 'erp':
                # MNE read_raw_egi
                # Suppress output
                raw = mne.io.read_raw_egi(path, preload=True, verbose=False)
                # Resample
                if raw.info['sfreq'] != self.target_rate:
                    raw.resample(self.target_rate, verbose=False)
                
                # Pick first 128 channels
                # EGI usually has E1..E128 plus others.
                raw.pick_channels([f'E{i}' for i in range(1, 129)])
                
                return raw.get_data() # (128, n_samples)
                
        except Exception as e:
            print(f"Error loading {modality} for {subject_id}: {e}")
            return None

    def create_dataset(self, modality='rest', window_size=4, stride=None):
        """
        Create dataset for modeling.
        Args:
            modality: 'rest' or 'erp'
            window_size: in seconds
            stride: in seconds (default: None, which means non-overlapping / stride=window_size)
        Returns:
            X: (n_segments, channels, time)
            y: (n_segments,)
            groups: (n_segments,)
        """
        X = []
        y = []
        groups = []
        
        samples_per_window = int(window_size * self.target_rate)
        if stride is None:
            step_size = samples_per_window
        else:
            step_size = int(stride * self.target_rate)
        
        for sub in self.common_subjects:
            # Get Label
            if modality == 'rest':
                label_str = self.meta_rest.loc[sub, 'type']
            else:
                label_str = self.meta_erp.loc[sub, 'type']
            
            # Binary Label
            if label_str == 'MDD':
                label = 1
            elif label_str == 'HC':
                label = 0
            else:
                continue
                
            # Load Data
            data = self.load_subject_data(sub, modality)
            if data is None: continue
            
            # Segment
            n_channels, n_points = data.shape
            
            # Overlapping windows
            for start in range(0, n_points - samples_per_window + 1, step_size):
                segment = data[:, start:start+samples_per_window]
                X.append(segment)
                y.append(label)
                groups.append(sub)
                
        return np.array(X), np.array(y), np.array(groups)

if __name__ == "__main__":
    # Test
    BASE_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA"
    REST_DIR = os.path.join(BASE_DIR, "EEG_128channels_resting_lanzhou_2015")
    ERP_DIR = os.path.join(BASE_DIR, "EEG_128channels_ERP_lanzhou_2015")
    EXCEL_REST = os.path.join(REST_DIR, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
    EXCEL_ERP = os.path.join(ERP_DIR, "subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx")
    
    loader = MODMADataLoader128(REST_DIR, ERP_DIR, EXCEL_REST, EXCEL_ERP)
    
    print("Testing Resting Load...")
    X_rest, y_rest, _ = loader.create_dataset('rest', window_size=2)
    print(f"Resting Dataset: {X_rest.shape}, {y_rest.shape}")
    
    print("Testing ERP Load...")
    X_erp, y_erp, _ = loader.create_dataset('erp', window_size=2)
    print(f"ERP Dataset: {X_erp.shape}, {y_erp.shape}")
