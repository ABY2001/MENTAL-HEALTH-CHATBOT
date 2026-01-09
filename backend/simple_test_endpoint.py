"""
Minimal test backend to isolate the issue
Run this instead of your main app to see detailed error messages
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
import traceback

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load model
print("\n" + "="*70)
print("Loading model...")
try:
    MODEL_PATH = "models/best_model.keras"
    ENCODER_PATH = "models/label_encoder.pkl"
    
    model = load_model(MODEL_PATH)
    print(f"✓ Model loaded")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder loaded")
    print(f"  Emotions: {list(label_encoder.classes_)}")
    
except Exception as e:
    print(f"✗ ERROR loading model:")
    traceback.print_exc()
    model = None
    label_encoder = None

print("="*70 + "\n")

@app.get("/")
def root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_input_shape": str(model.input_shape) if model else None
    }

@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile = File(...)):
    """Test endpoint with maximum error reporting"""
    
    print("\n" + "="*70)
    print("📥 RECEIVED REQUEST")
    print("="*70)
    
    if model is None:
        print("✗ Model not loaded!")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    temp_file = f"temp_{file.filename}"
    
    try:
        # Step 1: Save file
        print("\n1️⃣ Saving uploaded file...")
        content = await file.read()
        print(f"   File size: {len(content)} bytes")
        print(f"   File type: {file.content_type}")
        
        with open(temp_file, "wb") as f:
            f.write(content)
        print(f"   ✓ Saved to: {temp_file}")
        
        # Step 2: Extract features
        print("\n2️⃣ Extracting features...")
        try:
            # Import here to catch import errors
            from audio_utils import extract_mel_spectrogram
            print("   ✓ audio_utils imported")
            
            features = extract_mel_spectrogram(temp_file)
            
            if features is None:
                print("   ✗ extract_mel_spectrogram returned None")
                raise HTTPException(status_code=400, detail="Feature extraction failed")
            
            print(f"   ✓ Features shape: {features.shape}")
            print(f"   ✓ Features dtype: {features.dtype}")
            print(f"   ✓ Features range: [{features.min():.3f}, {features.max():.3f}]")
            
        except ImportError as e:
            print(f"   ✗ Import error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Import error: {str(e)}")
        except Exception as e:
            print(f"   ✗ Feature extraction error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Feature extraction error: {str(e)}")
        
        # Step 3: Prepare for model
        print("\n3️⃣ Preparing input for model...")
        features = np.expand_dims(features, axis=0)
        print(f"   After batch dim: {features.shape}")
        
        # Check model input shape
        model_input_dims = len(model.input_shape)
        print(f"   Model expects {model_input_dims}D input: {model.input_shape}")
        
        if model_input_dims == 4:
            features = np.expand_dims(features, axis=-1)
            print(f"   After channel dim: {features.shape}")
        
        # Verify shape matches
        expected_shape = model.input_shape
        actual_shape = features.shape
        
        print(f"   Expected: {expected_shape}")
        print(f"   Actual: {actual_shape}")
        
        # Check each dimension
        for i in range(1, len(expected_shape)):  # Skip batch dimension
            if expected_shape[i] is not None and expected_shape[i] != actual_shape[i]:
                error_msg = f"Shape mismatch at dimension {i}: expected {expected_shape[i]}, got {actual_shape[i]}"
                print(f"   ✗ {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
        
        print("   ✓ Shape validation passed")
        
        # Step 4: Predict
        print("\n4️⃣ Making prediction...")
        try:
            predictions = model.predict(features, verbose=1)[0]
            print(f"   ✓ Predictions shape: {predictions.shape}")
            print(f"   ✓ Predictions: {predictions}")
        except Exception as e:
            print(f"   ✗ Prediction error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
        # Step 5: Get emotion
        print("\n5️⃣ Decoding emotion...")
        idx = np.argmax(predictions)
        emotion = label_encoder.inverse_transform([idx])[0]
        confidence = float(predictions[idx])
        
        print(f"   ✓ Predicted index: {idx}")
        print(f"   ✓ Emotion: {emotion}")
        print(f"   ✓ Confidence: {confidence:.3f}")
        
        all_emotions = {
            label_encoder.inverse_transform([i])[0]: float(predictions[i])
            for i in range(len(predictions))
        }
        
        print("\n✅ SUCCESS!")
        print("="*70 + "\n")
        
        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "all_emotions": all_emotions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR:")
        traceback.print_exc()
        print("="*70 + "\n")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"🗑️  Cleaned up: {temp_file}")
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting minimal test backend...")
    print("📍 http://127.0.0.1:8000")
    print("📚 http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)