"""
Test script to diagnose audio processing issues
Run this to check if your model and audio processing work correctly
"""

import numpy as np
import pickle
from tensorflow.keras.models import load_model
import sys
import os

print("="*70)
print("🔍 DIAGNOSTIC TEST FOR AUDIO EMOTION DETECTION")
print("="*70)

# Test 1: Check if model files exist
print("\n1️⃣ Checking model files...")
MODEL_PATH = "models/best_model.keras"
ENCODER_PATH = "models/label_encoder.pkl"

if os.path.exists(MODEL_PATH):
    print(f"   ✓ Model found: {MODEL_PATH}")
else:
    print(f"   ✗ Model NOT found: {MODEL_PATH}")
    print(f"   Current directory: {os.getcwd()}")
    sys.exit(1)

if os.path.exists(ENCODER_PATH):
    print(f"   ✓ Encoder found: {ENCODER_PATH}")
else:
    print(f"   ✗ Encoder NOT found: {ENCODER_PATH}")
    sys.exit(1)

# Test 2: Load model
print("\n2️⃣ Loading model...")
try:
    model = load_model(MODEL_PATH)
    print(f"   ✓ Model loaded successfully")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load label encoder
print("\n3️⃣ Loading label encoder...")
try:
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"   ✓ Label encoder loaded")
    print(f"   Emotions: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"   ✗ Error loading encoder: {e}")
    sys.exit(1)

# Test 4: Check audio_utils
print("\n4️⃣ Testing audio_utils...")
try:
    from audio_utils import extract_mel_spectrogram
    print(f"   ✓ audio_utils imported successfully")
except Exception as e:
    print(f"   ✗ Error importing audio_utils: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create dummy audio features
print("\n5️⃣ Testing with dummy data...")
try:
    # Create dummy mel-spectrogram (64, 100)
    dummy_features = np.random.randn(64, 100).astype(np.float32)
    print(f"   Created dummy features: {dummy_features.shape}")
    
    # Prepare for model (add batch dimension)
    features = np.expand_dims(dummy_features, axis=0)
    print(f"   After adding batch dim: {features.shape}")
    
    # Check if we need channel dimension
    if len(model.input_shape) == 4:
        features = np.expand_dims(features, axis=-1)
        print(f"   After adding channel dim: {features.shape}")
    
    # Predict
    predictions = model.predict(features, verbose=0)[0]
    print(f"   ✓ Prediction successful!")
    print(f"   Predictions shape: {predictions.shape}")
    
    idx = np.argmax(predictions)
    emotion = label_encoder.inverse_transform([idx])[0]
    confidence = predictions[idx]
    
    print(f"   Predicted emotion: {emotion}")
    print(f"   Confidence: {confidence:.3f}")
    
except Exception as e:
    print(f"   ✗ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check dependencies
print("\n6️⃣ Checking dependencies...")
dependencies = {
    'librosa': None,
    'numpy': None,
    'tensorflow': None,
    'fastapi': None,
    'uvicorn': None
}

for dep in dependencies.keys():
    try:
        __import__(dep)
        print(f"   ✓ {dep} installed")
    except ImportError:
        print(f"   ✗ {dep} NOT installed")

# Test 7: Test actual audio file (if exists)
print("\n7️⃣ Testing with actual audio file...")
test_audio = "temp_recording.webm"
if os.path.exists(test_audio):
    print(f"   Found test audio: {test_audio}")
    try:
        features = extract_mel_spectrogram(test_audio)
        if features is not None:
            print(f"   ✓ Features extracted: {features.shape}")
            
            # Prepare for model
            features_model = np.expand_dims(features, axis=0)
            if len(model.input_shape) == 4:
                features_model = np.expand_dims(features_model, axis=-1)
            
            predictions = model.predict(features_model, verbose=0)[0]
            idx = np.argmax(predictions)
            emotion = label_encoder.inverse_transform([idx])[0]
            confidence = predictions[idx]
            
            print(f"   ✓ Prediction from audio: {emotion} ({confidence:.3f})")
        else:
            print(f"   ✗ Failed to extract features")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"   No test audio file found (this is okay)")

print("\n" + "="*70)
print("✅ DIAGNOSTIC TEST COMPLETE")
print("="*70)
print("\nIf all tests passed, your setup is correct!")
print("If you see errors, check the messages above for details.")
print("\nExpected model input shape:")
print("  - 3D: (None, 64, 100) for mel-spectrogram")
print("  - 4D: (None, 64, 100, 1) for mel-spectrogram with channel")
print("\n" + "="*70)