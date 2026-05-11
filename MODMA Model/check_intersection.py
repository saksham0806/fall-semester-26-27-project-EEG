import os
from glob import glob
import pandas as pd

REST_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA/EEG_128channels_resting_lanzhou_2015"
ERP_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA/EEG_128channels_ERP_lanzhou_2015"

def extract_ids(directory, extension):
    files = glob(os.path.join(directory, f"*{extension}"))
    ids = set()
    for f in files:
        basename = os.path.basename(f)
        # Extract ID: 02010002... -> 2010002 (matches excel)
        # Filenames start with '0', excel IDs don't.
        # Let's extract the first numeric sequence.
        
        # Example: 02010002rest... 
        # We need to handle this robustly.
        id_part = basename[:8] # 02010002
        
        # Remove leading zero to match Excel if needed, or keep as canonical ID
        if id_part.startswith('0'):
            id_part = id_part[1:]
        
        # Some might be 202... 
        ids.add(id_part)
    return ids

rest_ids = extract_ids(REST_DIR, ".mat")
erp_ids = extract_ids(ERP_DIR, ".raw")

print(f"Resting Subjects: {len(rest_ids)}")
print(f"ERP Subjects: {len(erp_ids)}")

intersection = rest_ids.intersection(erp_ids)
print(f"Intersection: {len(intersection)}")
print("Sample Intersection:", list(intersection)[:5])
