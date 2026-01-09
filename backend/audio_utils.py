import librosa
import numpy as np
import soundfile as sf

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
        
        # Try loading with librosa (handles most formats via audioread/soundfile)
        try:
            y, sr = librosa.load(file_path, sr=16000, duration=5.0)
            print(f"✓ Audio loaded with librosa: duration={len(y)/sr:.2f}s, sr={sr}Hz")
        except Exception as e:
            print(f"Librosa failed, trying soundfile: {e}")
            # Fallback to soundfile
            y, sr = sf.read(file_path)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            print(f"✓ Audio loaded with soundfile: duration={len(y)/sr:.2f}s, sr={sr}Hz")
        
        # Check if audio is empty or too short
        if len(y) == 0:
            print("✗ Error: Audio is empty")
            return None
        
        print(f"Audio samples: {len(y)}, duration: {len(y)/sr:.2f}s")
        
        # Remove silence from beginning and end (but not too aggressively)
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            if len(y_trimmed) > sr * 0.3:  # Keep if at least 0.3 seconds
                y = y_trimmed
                print(f"After trimming: duration={len(y)/sr:.2f}s")
            else:
                print("Trimming removed too much, using original")
        except:
            print("Trimming failed, using original audio")
        
        # Ensure minimum length
        min_length = int(sr * 0.5)  # 0.5 seconds minimum
        if len(y) < min_length:
            print(f"Audio too short ({len(y)/sr:.2f}s), padding to 0.5s")
            y = np.pad(y, (0, min_length - len(y)), mode='constant')
        
        # Normalize audio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        print(f"Final audio length: {len(y)/sr:.2f}s")
        
        # Extract mel-spectrogram with error handling
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=n_mels,
                n_fft=1024,
                hop_length=512,
                fmax=8000,
                window='hann'
            )
            print(f"Mel-spectrogram computed: {mel_spec.shape}")
        except Exception as e:
            print(f"✗ Error computing mel-spectrogram: {e}")
            return None
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Check for invalid values
        if np.isnan(mel_spec_db).any() or np.isinf(mel_spec_db).any():
            print("✗ Error: Mel-spectrogram contains NaN or Inf values")
            return None
        
        # Normalize to zero mean and unit variance
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / std
        else:
            print("Warning: std is zero, skipping normalization")
        
        print(f"Normalized mel-spectrogram: shape={mel_spec_db.shape}, range=[{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}]")
        
        # Pad or truncate to fixed length
        current_len = mel_spec_db.shape[1]
        if current_len < max_len:
            pad_width = max_len - current_len
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            print(f"Padded from {current_len} to {max_len} frames")
        elif current_len > max_len:
            mel_spec_db = mel_spec_db[:, :max_len]
            print(f"Truncated from {current_len} to {max_len} frames")
        
        print(f"✓ Final mel-spectrogram shape: {mel_spec_db.shape}")
        
        return mel_spec_db.astype(np.float32)
        
    except Exception as e:
        print(f"✗ Error in extract_mel_spectrogram: {e}")
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
        y, sr = librosa.load(file_path, sr=16000, duration=3.0)
        
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
        
        return mfcc.astype(np.float32)
        
    except Exception as e:
        print(f"Error in extract_mfcc_features: {e}")
        import traceback
        traceback.print_exc()
        return None