import librosa
import numpy as np

def extract_mel_spectrogram(file_path, n_mels=64, max_len=100):
    """
    Extract mel-spectrogram features from audio file
    Supports multiple audio formats including webm, wav, mp3
    
    Args:
        file_path: Path to audio file
        n_mels: Number of mel bands (default: 64)
        max_len: Maximum length of spectrogram (default: 100)
    
    Returns:
        mel_spec_db: Mel-spectrogram in dB scale, normalized
    """
    try:
        print(f"Loading audio from: {file_path}")
        
        # Load audio with librosa (handles webm, wav, mp3, etc.)
        y, sr = librosa.load(file_path, sr=16000, duration=2.5)
        
        print(f"Audio loaded: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
        
        # Remove silence from beginning and end
        y, _ = librosa.effects.trim(y, top_db=20)
        
        print(f"After trimming: duration={len(y)/sr:.2f}s")
        
        # Check if audio is too short
        if len(y) < sr * 0.5:  # Less than 0.5 seconds
            print("Warning: Audio too short, padding...")
            y = np.pad(y, (0, sr - len(y)), mode='constant')
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512,
            fmax=8000
        )
        
        print(f"Mel-spectrogram shape before dB conversion: {mel_spec.shape}")
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to zero mean and unit variance
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        print(f"Mel-spectrogram shape after normalization: {mel_spec_db.shape}")
        
        # Pad or truncate to fixed length
        if mel_spec_db.shape[1] < max_len:
            pad_width = max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            print(f"Padded to {max_len} frames")
        else:
            mel_spec_db = mel_spec_db[:, :max_len]
            print(f"Truncated to {max_len} frames")
        
        print(f"Final mel-spectrogram shape: {mel_spec_db.shape}")
        
        return mel_spec_db
        
    except Exception as e:
        print(f"Error in extract_mel_spectrogram: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_mfcc_features(file_path, n_mfcc=40, max_len=128):
    """
    Extract MFCC features from audio file (alternative feature extraction)
    
    Args:
        file_path: Path to audio file
        n_mfcc: Number of MFCC coefficients (default: 40)
        max_len: Maximum length of sequence (default: 128)
    
    Returns:
        mfcc: MFCC features
    """
    try:
        print(f"Extracting MFCC from: {file_path}")
        
        # Load audio
        y, sr = librosa.load(file_path, sr=16000, duration=2.5)
        
        # Remove silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Transpose to (time_steps, features)
        mfcc = mfcc.T
        
        print(f"MFCC shape: {mfcc.shape}")
        
        # Pad or truncate
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len, :]
        
        print(f"Final MFCC shape: {mfcc.shape}")
        
        return mfcc
        
    except Exception as e:
        print(f"Error in extract_mfcc_features: {e}")
        import traceback
        traceback.print_exc()
        return None