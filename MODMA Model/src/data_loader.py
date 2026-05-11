import os
import numpy as np
import pandas as pd
from glob import glob

class MODMADataLoader:
    def __init__(self, dataset_path, excel_path, sampling_rate=250):
        """
        Initialize the data loader.
        
        Args:
            dataset_path (str): Path to the directory containing .txt EEG files.
            excel_path (str): Path to the subject information Excel file.
            sampling_rate (int): Sampling rate of the EEG data (default 250Hz).
        """
        self.dataset_path = dataset_path
        self.excel_path = excel_path
        self.sampling_rate = sampling_rate
        self.subjects_df = None
        self.file_map = {} # Map subject_id -> file_path
        
        self._load_metadata()
        self._map_files()

    def _load_metadata(self):
        """Load subject information and labels."""
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel file not found at {self.excel_path}")
        
        # Load excel, skipping header if necessary (based on previous view, headers are on row 0)
        self.subjects_df = pd.read_excel(self.excel_path)
        
        # Clean up column names (strip whitespace)
        self.subjects_df.columns = self.subjects_df.columns.str.strip()
        
        # Ensure 'subject id' is treated as string for matching
        if 'subject id' in self.subjects_df.columns:
            self.subjects_df['subject id'] = self.subjects_df['subject id'].astype(str)
            
        print(f"Loaded metadata for {len(self.subjects_df)} subjects.")

    def _map_files(self):
        """Map subject IDs to their corresponding .txt data files."""
        # Pattern to match: subject IDs are usually part of the filename, e.g., 02010001_still.txt
        # The Excel has IDs like 2010001. The files have prefix '0'.
        # Let's inspect the files and finding matching strategy.
        
        files = glob(os.path.join(self.dataset_path, "*_still.txt"))
        
        for file_path in files:
            basename = os.path.basename(file_path)
            # Extract ID from filename (e.g., 02010001 -> 2010001 maybe?)
            # Let's try to match loosely.
            # Filename: 02010001_still.txt -> ID 02010001
            # Excel ID: 2010002 -> Filename 02010002? 
            # Currently assuming direct containment or slight prefix difference.
            
            file_id = basename.split('_')[0]
            
            # The excel IDs are like 2010002 (int). The files are 02010002.
            # We need to handle the leading zero or type mismatch.
            
            self.file_map[file_id] = file_path

        print(f"Mapped {len(self.file_map)} data files.")

    def get_subject_data(self, subject_id):
        """
        Load raw data for a specific subject.
        
        Returns:
            np.array: Shape (n_samples, n_channels)
        """
        # Normalize subject_id to string and handle leading zeros if necessary
        subject_id_str = str(subject_id)
        
        # Try exact match first
        if subject_id_str in self.file_map:
            path = self.file_map[subject_id_str]
        # Try adding a leading zero (Excel: 2010001 -> File: 02010001)
        elif f"0{subject_id_str}" in self.file_map:
             path = self.file_map[f"0{subject_id_str}"]
        else:
            print(f"Warning: No file found for subject {subject_id}")
            return None
            
        # Load txt file
        # The file format (based on view) is space-separated values, no header
        # 3 columns corresponding to 3 channels.
        try:
            data = pd.read_csv(path, sep=r'\s+', header=None)
            return data.values
        except Exception as e:
            print(f"Error loading file for {subject_id}: {e}")
            return None

    def load_dataset(self, segment_length=2000, overlap=0, filter_labels=['MDD', 'HC']):
        """
        Load and segment data for all matching subjects.
        
        Args:
            segment_length (int): Number of samples per segment.
            overlap (int): Overlap between segments (samples).
            filter_labels (list): Types to include.
            
        Returns:
            X (np.array): Segments (n_segments, segment_length, n_channels)
            y (np.array): Labels (n_segments,)
            groups (np.array): Subject IDs for GroupKFold (n_segments,)
        """
        X_list = []
        y_list = []
        groups_list = []
        
        for _, row in self.subjects_df.iterrows():
            sub_id = row['subject id']
            label_str = row['type']
            
            if label_str not in filter_labels:
                continue
                
            # Convert label to binary: MDD=1, HC=0 (Assumes only these two exist or are filtered)
            label = 1 if label_str == 'MDD' else 0
            
            raw_data = self.get_subject_data(sub_id)
            if raw_data is None:
                continue
                
            # Segmentation
            n_samples = raw_data.shape[0]
            step = segment_length - overlap
            
            for start in range(0, n_samples - segment_length + 1, step):
                end = start + segment_length
                segment = raw_data[start:end, :]
                
                # Basic check for flat lines or artifacts could go here
                
                X_list.append(segment)
                y_list.append(label)
                groups_list.append(sub_id)
                
        X = np.array(X_list)
        y = np.array(y_list)
        groups = np.array(groups_list)
        
        print(f"Created dataset: X shape={X.shape}, y shape={y.shape}")
        return X, y, groups

if __name__ == "__main__":
    # Test execution
    DATA_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA/EEG_3channels_resting_lanzhou_2015"
    EXCEL_FILE = os.path.join(DATA_DIR, "subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx")
    
    loader = MODMADataLoader(DATA_DIR, EXCEL_FILE)
    if loader.subjects_df is not None:
        print("First 5 subjects from Excel:")
        print(loader.subjects_df.head())
        
        # Test loading one subject
        first_sub = loader.subjects_df.iloc[0]['subject id']
        data = loader.get_subject_data(first_sub)
        if data is not None:
            print(f"Loaded data for {first_sub}: shape {data.shape}")
            
        # Test full load (small segment for speed)
        X, y, groups = loader.load_dataset(segment_length=250*4, overlap=0)
