import numpy as np
from scipy.signal import welch

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 45)
}

def compute_psd_features(data, fs=128):
    """
    Compute PSD features for EEG data.
    Args:
        data: (n_channels, n_samples)
        fs: Sampling frequency
    Returns:
        features: (n_channels * n_bands, ) flattened feature vector
    """
    n_channels = data.shape[0]
    features = []
    
    # Compute PSD using Welch's method
    # nperseg=fs (1 sec window) provides 1Hz resolution
    freqs, psd = welch(data, fs=fs, nperseg=fs*2)
    
    for _, (low, high) in BANDS.items():
        # Find indices of band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Mean PSD in band
        band_power = np.mean(psd[:, idx_band], axis=1) # (n_channels,)
        features.append(band_power)
        
    return np.concatenate(features)

def compute_de_features(data, fs=128):
    """
    Compute Differential Entropy (DE) features.
    DE for Gaussian distribution is equivalent to log(PSD).
    """
    n_channels = data.shape[0]
    features = []
    
    freqs, psd = welch(data, fs=fs, nperseg=fs*2)
    
    for _, (low, high) in BANDS.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.mean(psd[:, idx_band], axis=1)
        # DE ~ log(power)
        de = np.log(band_power + 1e-10) 
        features.append(de)
        
    return np.concatenate(features)

def extract_features(X, fs=128, feature_type='psd'):
    """
    Batch extraction.
    Args:
        X: (n_segments, n_channels, n_samples)
    Returns:
        X_feats: (n_segments, n_features)
    """
    features_list = []
    for i in range(X.shape[0]):
        if feature_type == 'psd':
            feat = compute_psd_features(X[i], fs)
        elif feature_type == 'de':
            feat = compute_de_features(X[i], fs)
        else:
            raise ValueError("Unknown feature type")
        features_list.append(feat)
        
    return np.array(features_list)

if __name__ == "__main__":
    # Test
    dummy_data = np.random.randn(10, 128, 512) # 10 segments
    feats = extract_features(dummy_data, fs=128, feature_type='psd')
    print(f"Features shape: {feats.shape}")
    # Expected: 10, 128*5 = 640
