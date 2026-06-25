import numpy as np
from scipy.signal import welch, butter, lfilter

# Version-compatible trapezoid integration for NumPy 1.x and 2.x
try:
    from scipy.integrate import trapezoid
except ImportError:
    try:
        from scipy.integrate import trapz as trapezoid
    except ImportError:
        try:
            from numpy import trapezoid
        except ImportError:
            from numpy import trapz as trapezoid


# Standard EEG frequency bands
BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (12.0, 30.0),
    'gamma': (30.0, 45.0)
}

# Channel index mapping for the 6-channel frontal setup
# Channels: Fp1, Fp2, F7, F3, F4, F8
CHANNEL_MAP = {
    'Fp1': 0,
    'Fp2': 1,
    'F7': 2,
    'F3': 3,
    'F4': 4,
    'F8': 5
}

# Pairs for Frontal Asymmetry (Right - Left)
ASYMMETRY_PAIRS = [
    ('Fp2', 'Fp1'),  # Prefrontal asymmetry
    ('F4', 'F3'),    # Mid-frontal asymmetry
    ('F8', 'F7')     # Lateral frontal asymmetry
]

def butter_bandpass(lowcut, highcut, fs, order=3):
    """Generate Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clip to avoid exceeding Nyquist frequency
    low = np.clip(low, 1e-5, 0.999)
    high = np.clip(high, 1e-5, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Apply Butterworth bandpass filter to signal data along the last axis."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=-1)

def extract_psd_features(X, fs=128, relative=False):
    """
    Extract Power Spectral Density (PSD) features for each channel and band.
    
    Args:
        X (np.ndarray): EEG segments of shape (N, n_channels, n_samples).
        fs (int): Sampling frequency in Hz.
        relative (bool): If True, return relative power instead of absolute.
        
    Returns:
        features (np.ndarray): PSD features of shape (N, n_channels * 5).
    """
    N, n_channels, n_samples = X.shape
    n_bands = len(BANDS)
    features = np.zeros((N, n_channels * n_bands))
    
    # Run Welch method over all segments
    # nperseg is set to 256 for a frequency resolution of 0.5Hz at 128Hz
    nperseg = min(256, n_samples)
    freqs, psd = welch(X, fs=fs, nperseg=nperseg, axis=-1)
    
    for b_idx, (band_name, (low, high)) in enumerate(BANDS.items()):
        # Find frequency indices within the band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        
        # Integrate PSD values within the band
        # shape of psd: (N, n_channels, n_freq_bins)
        band_power = trapezoid(psd[:, :, idx_band], freqs[idx_band], axis=-1)
        
        if relative:
            total_power = trapezoid(psd, freqs, axis=-1) + 1e-8
            band_power = band_power / total_power
            
        features[:, b_idx * n_channels : (b_idx + 1) * n_channels] = band_power
        
    return features

def extract_differential_entropy(X, fs=128):
    """
    Extract Differential Entropy (DE) features for each channel and band.
    DE = 0.5 * ln(2 * pi * e * sigma^2)
    
    Args:
        X (np.ndarray): EEG segments of shape (N, n_channels, n_samples).
        fs (int): Sampling frequency in Hz.
        
    Returns:
        features (np.ndarray): DE features of shape (N, n_channels * 5).
    """
    N, n_channels, n_samples = X.shape
    n_bands = len(BANDS)
    features = np.zeros((N, n_channels * n_bands))
    
    constant = 0.5 * np.log(2 * np.pi * np.e)
    
    for b_idx, (band_name, (low, high)) in enumerate(BANDS.items()):
        # Bandpass filter the raw signals
        filtered_X = butter_bandpass_filter(X, low, high, fs, order=3)
        
        # Calculate variance along the time axis (axis=-1)
        # Add epsilon to prevent log of 0
        var = np.var(filtered_X, axis=-1) + 1e-8
        
        # Compute Differential Entropy
        de = 0.5 * np.log(var) + constant
        
        features[:, b_idx * n_channels : (b_idx + 1) * n_channels] = de
        
    return features

def extract_faa_features(X, fs=128):
    """
    Extract Frontal Alpha Asymmetry (FAA) features for pre-defined frontal pairs.
    FAA = ln(Alpha_right) - ln(Alpha_left)
    
    Args:
        X (np.ndarray): EEG segments of shape (N, n_channels, n_samples).
        fs (int): Sampling frequency in Hz.
        
    Returns:
        features (np.ndarray): FAA features of shape (N, n_pairs).
    """
    N, n_channels, n_samples = X.shape
    n_pairs = len(ASYMMETRY_PAIRS)
    features = np.zeros((N, n_pairs))
    
    # Calculate Welch PSD to isolate Alpha band power
    nperseg = min(256, n_samples)
    freqs, psd = welch(X, fs=fs, nperseg=nperseg, axis=-1)
    
    # Isolate Alpha band
    low, high = BANDS['alpha']
    idx_alpha = np.logical_and(freqs >= low, freqs <= high)
    alpha_power = trapezoid(psd[:, :, idx_alpha], freqs[idx_alpha], axis=-1)
    
    # Add epsilon to avoid log of 0
    alpha_power = np.maximum(alpha_power, 1e-8)
    log_alpha = np.log(alpha_power)
    
    for p_idx, (right_ch, left_ch) in enumerate(ASYMMETRY_PAIRS):
        right_idx = CHANNEL_MAP[right_ch]
        left_idx = CHANNEL_MAP[left_ch]
        
        # FAA = ln(Right) - ln(Left)
        features[:, p_idx] = log_alpha[:, right_idx] - log_alpha[:, left_idx]
        
    return features

def extract_spectral_entropy(X, fs=128):
    """
    Extract Spectral Entropy for each channel.
    Spectral Entropy is the Shannon entropy of the normalized PSD.
    """
    N, n_channels, n_samples = X.shape
    nperseg = min(256, n_samples)
    freqs, psd = welch(X, fs=fs, nperseg=nperseg, axis=-1)
    
    # Normalize PSD per segment and channel to get a probability distribution
    psd_sum = np.sum(psd, axis=-1, keepdims=True) + 1e-8
    p = psd / psd_sum
    
    # Calculate Shannon Entropy (using log base 2)
    entropy = -np.sum(p * np.log2(p + 1e-12), axis=-1)
    return entropy

def extract_amplitude_features(X):
    """
    Extract Root Mean Square (RMS) and Standard Deviation (STD) for each channel.
    """
    # STD
    std_feat = np.std(X, axis=-1)
    
    # RMS
    rms_feat = np.sqrt(np.mean(X**2, axis=-1))
    
    return np.hstack((std_feat, rms_feat))

def extract_all_features(X, fs=128):
    """
    Extract and concatenate all features (Relative PSD, DE, FAA, Spectral Entropy, RMS, and STD).
    For 6 channels, shape will be (N, 30 + 30 + 3 + 6 + 12) = (N, 81) features.
    """
    psd_feat = extract_psd_features(X, fs=fs, relative=True)
    de_feat = extract_differential_entropy(X, fs=fs)
    faa_feat = extract_faa_features(X, fs=fs)
    se_feat = extract_spectral_entropy(X, fs=fs)
    amp_feat = extract_amplitude_features(X)
    
    return np.hstack((psd_feat, de_feat, faa_feat, se_feat, amp_feat))
