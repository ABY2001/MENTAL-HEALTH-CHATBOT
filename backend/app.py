# app.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
import logging
from dotenv import load_dotenv

load_dotenv()

from database import Base, engine, SessionLocal
from models import User
from audio_utils import extract_mel_spectrogram
from safety_engine import SafetyTriageEngine

# ⭐ NEW IMPORT: Use the unified service
from groq_service import ai

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Mental Health Support API")
logging.getLogger("uvicorn.access").disabled = True

# ==================== INITIALIZATION ====================
print("\n" + "="*70)
print("🚀 SYSTEM STARTUP")
print("="*70)

safety_engine = SafetyTriageEngine()
print("✓ Safety Engine initialized")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Load Audio Models
MODEL_PATH = "models/best_model.keras"
ENCODER_PATH = "models/label_encoder.pkl"
model = None
label_encoder = None

try:
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("✓ Local Audio Model loaded")
except:
    print("⚠️ Local Audio Model NOT loaded")

# Data Models
class LoginRequest(BaseModel):
    email: str
    password: str

class TextRequest(BaseModel):
    text: str

# Database
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ==================== ENDPOINTS ====================

@app.get("/")
def health_check():
    return {"status": "online", "ai_connected": ai.available}

@app.post("/predict-emotion-text")
def predict_emotion_text(request: TextRequest):
    text = request.text.strip()
    if not text: raise HTTPException(status_code=400, detail="Empty text")
    
    print(f"\n📨 Received: '{text}'")
    
    # 1. Detect Emotion (via Gemini)
    emotion, confidence, _ = ai.detect_emotion(text)
    print(f"   Emotion: {emotion} ({confidence})")
    
    # 2. Safety Check
    safety = safety_engine.evaluate(text, emotion, confidence)
    
    # 3. Generate Response (via Gemini)
    bot_response = ai.chat(text, emotion, safety.crisis_detected)
    
    # 4. Fallback if Gemini fails
    if not bot_response:
        print("   ⚠️ Gemini Failed -> Using Fallback")
        bot_response = "I'm here to listen. Please tell me more about what you're going through."
    else:
        print(f"   Reply: {bot_response}")

    return {
        "emotion": emotion,
        "confidence": confidence,
        "bot_response": bot_response,
        "safety": {
            "risk_level": safety.risk_level.value,
            "crisis_detected": safety.crisis_detected
        }
    }

# Audio Endpoint
@app.post("/predict-emotion")
def predict_emotion(file: UploadFile = File(...)):
    if model is None: raise HTTPException(status_code=500, detail="Audio model missing")
    
    temp_file = f"temp_{file.filename}"
    try:
        content = file.file.read()
        with open(temp_file, "wb") as f: f.write(content)
        
        # Audio Feature Extraction
        features = extract_mel_spectrogram(temp_file)
        if features is None: raise HTTPException(status_code=400, detail="Audio error")
        
        # Audio Prediction
        features_input = np.expand_dims(features, axis=0)
        if len(model.input_shape) == 4: features_input = np.expand_dims(features_input, axis=-1)
        predictions = model.predict(features_input, verbose=0)[0]
        idx = np.argmax(predictions)
        emotion = label_encoder.inverse_transform([idx])[0]
        confidence = float(predictions[idx])
        
        # Generate Response based on audio emotion
        safety = safety_engine.evaluate("voice message", emotion, confidence)
        bot_response = ai.chat("Voice message sent", emotion, safety.crisis_detected) or "I'm listening."
        
        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "bot_response": bot_response,
            "safety": {"crisis_detected": safety.crisis_detected}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file): os.remove(temp_file)

# Auth Endpoints
@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email.lower()).first()
    if not user or request.password != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "user_id": user.id}

@app.post("/register")
def register(request: LoginRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == request.email.lower()).first():
        raise HTTPException(status_code=400, detail="Email exists")
    new_user = User(email=request.email.lower(), password=request.password)
    db.add(new_user)
    db.commit()
    return {"message": "Registered", "user_id": new_user.id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)